"""Adaptive Patch Transformer in space (APT-2D) for satellite image time series.

Choudhury et al., "Accelerating Vision Transformers with Adaptive Patch Sizes" (APT), arXiv 2510.18091.
APT splits an image coarse to fine (quadtree): a region whose pixel entropy is below the threshold
of its scale becomes one big token, otherwise it is split into 4 smaller regions.

Our version for time series:
    sizes    p0, 2 p0, 4 p0, ... (model.patch_size = p0, one threshold per larger size)
    one partition per sample, shared by all dates: entropy is averaged over the real dates
             (field borders do not move during a season; time is handled separately, e.g. RLT)
    entropy  Shannon entropy of the normalised band values in the region (all bands and pixels,
             clipped to [-3, 3], 32 bins), as in APT's H(P) = -sum p_i log2 p_i
    embed    E(big) = E(avgpool(big) to p0 x p0) + ZeroLinear(conv-merged p0 embeddings)   (APT eq.)
    position base-grid encoding averaged over the cells a big token covers
    output   each token is copied to every p0 x p0 cell it covers, then the usual pixel head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BINS, LO, HI = 32, -3.0, 3.0


@torch.no_grad()
def region_entropy(img: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
    """img (B, T, C, H, W) normalised, mask (B, T). Returns (B, H/size, W/size): entropy in bits
    of each size x size region, averaged over the real dates."""
    b, t, c, h, w = img.shape
    gh, gw = h // size, w // size
    q = ((img.float().clamp(LO, HI - 1e-6) - LO) / (HI - LO) * BINS).long()          # (B, T, C, H, W)
    q = q.reshape(b, t, c, gh, size, gw, size).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t, gh * gw, -1)
    counts = torch.zeros(b, t, gh * gw, BINS, device=img.device).scatter_add_(3, q, torch.ones_like(q, dtype=torch.float))
    p = counts / counts.sum(-1, keepdim=True)
    ent = -(p * torch.log2(p.clamp(min=1e-12))).sum(-1)                               # (B, T, R)
    m = mask.float()[:, :, None]
    ent = (ent * m).sum(1) / m.sum(1).clamp(min=1)
    return ent.view(b, gh, gw)


@torch.no_grad()
def quadtree(img: torch.Tensor, mask: torch.Tensor, p0: int, taus: list[float]) -> list[torch.Tensor]:
    """Coarse-to-fine split. taus[i] is the threshold for size p0 * 2**(i+1).
    Returns one bool map per level, finest first: sel[l] (B, H/(p0 2^l), W/(p0 2^l)), True = token
    of that size there. Every pixel is covered by exactly one token."""
    L = len(taus)
    sel = [None] * (L + 1)
    open_ = None                                          # regions not yet assigned at this level
    for lvl in range(L, 0, -1):
        s = p0 * 2 ** lvl
        low = region_entropy(img, mask, s) < taus[lvl - 1]
        if open_ is None:
            open_ = torch.ones_like(low)
        sel[lvl] = open_ & low
        open_ = (open_ & ~low).repeat_interleave(2, 1).repeat_interleave(2, 2)
    sel[0] = open_ if open_ is not None else torch.ones(img.shape[0], img.shape[-2] // p0, img.shape[-1] // p0,
                                                         dtype=torch.bool, device=img.device)
    return sel


def expand(x: torch.Tensor, f: int) -> torch.Tensor:
    """(B, gh, gw, ...) -> (B, gh f, gw f, ...): copy each cell to f x f cells."""
    return x.repeat_interleave(f, 1).repeat_interleave(f, 2)


class APTEmbed(nn.Module):
    """Token features for every level on a dense grid; the model then keeps the selected ones."""

    def __init__(self, in_ch: int, dim: int, p0: int, levels: int):
        super().__init__()
        self.p0, self.levels = p0, levels
        self.proj = nn.Conv2d(in_ch, dim, p0, p0)                                   # base embedding E
        self.merge = nn.ModuleList(nn.Conv2d(dim, dim, 2, 2) for _ in range(levels))  # 2x2 -> 1, per level
        self.zero = nn.ModuleList(nn.Linear(dim, dim) for _ in range(levels))
        for z in self.zero:                                                         # ZeroMLP (APT, ControlNet)
            nn.init.zeros_(z.weight)
            nn.init.zeros_(z.bias)

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        """img (B, T, C, H, W) -> list per level of (B, T, gh_l, gw_l, D), finest first."""
        b, t, c, h, w = img.shape
        x = img.reshape(b * t, c, h, w)
        base = self.proj(x)                                                         # (BT, D, h0, w0)
        out = [base]
        merged = base
        for lvl in range(1, self.levels + 1):
            f = 2 ** lvl
            resized = self.proj(F.avg_pool2d(x, f))                                 # E(Resize_p0(P))
            merged = self.merge[lvl - 1](merged)                                    # conv stack, applied lvl times
            z = self.zero[lvl - 1](merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out.append(resized + z)
        return [o.permute(0, 2, 3, 1).reshape(b, t, o.shape[-2], o.shape[-1], -1) for o in out]


def pool_pe(pe: torch.Tensor | None, gh: int, gw: int, levels: int) -> list[torch.Tensor | None]:
    """Base grid encoding (gh*gw, D) -> list per level of (gh_l, gw_l, D), averaged over covered cells."""
    if pe is None:
        return [None] * (levels + 1)
    g = pe.view(gh, gw, -1).permute(2, 0, 1)[None]
    return [F.avg_pool2d(g, 2 ** l)[0].permute(1, 2, 0) if l else g[0].permute(1, 2, 0) for l in range(levels + 1)]

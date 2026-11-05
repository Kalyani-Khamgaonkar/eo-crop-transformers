"""New patchification and positional-encoding options for the 9-configuration study.

Patchification
    groups   SpectralGroupTokenizer: one token per spectral group per date and patch    (SatMAE)
             RGB+NIR = B2 B3 B4 B8, red edge = B5 B6 B7 B8A, SWIR = B11 B12

How position enters the token (model.pe_mode)
    add      learnable / sinusoidal spatial PE + temporal encoding, added             (TSViT; existing code)
    concat   fixed sin-cos of (t, y, x[, g]) concatenated to D dims, then added        (SatMAE)
    conv     no vector added; depth-wise 3D conv over (t, y, x) inside every MLP       (VistaFormer Mix-FFN)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.models.pos_encoding import sincos_1d

GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))   # band order B2 B3 B4 B5 B6 B7 B8 B8A B11 B12


class SpectralGroupTokenizer(nn.Module):
    """x (B, T, C, H, W) -> tokens (B, T, G*N, D), group-major then row-major grid."""

    def __init__(self, bands: int, dim: int, p: int, groups=GROUPS):
        super().__init__()
        assert sorted(i for g in groups for i in g) == list(range(bands)), "groups must cover every band once"
        self.p, self.groups = p, [list(g) for g in groups]
        self.proj = nn.ModuleList(nn.Linear(len(g) * p * p, dim) for g in self.groups)

    def forward(self, x, doy, mask):
        b, t, c, h, w = x.shape
        p, gh, gw = self.p, h // self.p, w // self.p
        out = []
        for g, proj in zip(self.groups, self.proj):
            xg = x[:, :, g].reshape(b, t, len(g), gh, p, gw, p)
            xg = xg.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t, gh * gw, len(g) * p * p)
            out.append(proj(xg))                                       # (B, T, N, D)
        return torch.cat(out, dim=2), doy, mask


def concat_sincos_pe(doy_t: torch.Tensor, gh: int, gw: int, groups: int, dim: int, group_sin: bool,
                     time: str = "date") -> torch.Tensor:
    """Concatenated sin-cos position. Returns (B, T', G*N, D).
    time = "date": real acquisition day (SatMAE); "index": frame number 0, 1, 2, ... (Prithvi, TiMo).
    Split: (t, y, x) = (D/2, D/4, D/4); with group_sin (t, y, x, g) = (D/4, D/4, D/4, D/4)."""
    b, t = doy_t.shape
    if time == "index":
        doy_t = torch.arange(t, device=doy_t.device).expand(b, t)
    n = gh * gw
    dt, dy = (dim // 4, dim // 4) if group_sin else (dim // 2, dim // 4)
    assert dt % 2 == 0 and dy % 2 == 0
    ys, xs = torch.meshgrid(torch.arange(gh, device=doy_t.device), torch.arange(gw, device=doy_t.device), indexing="ij")
    yx = torch.cat([sincos_1d(ys.flatten(), dy), sincos_1d(xs.flatten(), dy)], dim=-1)      # (N, 2*dy)
    yx = yx.repeat(groups, 1)                                                             # (G*N, 2*dy)
    parts = [sincos_1d(doy_t, dt, max_period=1000.0)[:, :, None, :].expand(b, t, groups * n, dt),
             yx[None, None].expand(b, t, groups * n, 2 * dy)]
    if group_sin:
        g = sincos_1d(torch.arange(groups, device=doy_t.device), dim - dt - 2 * dy).repeat_interleave(n, dim=0)
        parts.append(g[None, None].expand(b, t, groups * n, g.shape[-1]))
    pe = torch.cat(parts, dim=-1)
    assert pe.shape[-1] == dim
    return pe


class ConvMLP(nn.Module):
    """Mix-FFN: fc1 -> depth-wise 3x3x3 conv over (t, y, x) -> GELU -> dropout -> fc2 -> dropout.
    Position comes from the conv's zero padding at the borders; padded dates are zeroed before the conv.
    Call set_grid(t, G, gh, gw, mask_t) before each forward (done by the model)."""

    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.dw = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
        self.grid, self.mask_t = None, None

    def set_grid(self, t, groups, gh, gw, mask_t):
        self.grid, self.mask_t = (t, groups, gh, gw), mask_t

    def forward(self, x):
        b, length, _ = x.shape
        t, g, gh, gw = self.grid
        h = self.fc1(x)
        hid = h.shape[-1]
        h = h.view(b, t, g, gh, gw, hid) * self.mask_t[:, :, None, None, None, None].to(h.dtype)
        h = h.permute(0, 2, 5, 1, 3, 4).reshape(b * g, hid, t, gh, gw)
        h = self.dw(h)
        h = h.view(b, g, hid, t, gh, gw).permute(0, 3, 1, 4, 5, 2).reshape(b, length, hid)
        return self.drop(self.fc2(self.drop(self.act(h))))

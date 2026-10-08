"""SO1: spatial-temporal patchification (turning the image cube into tokens).

All tokenizers take
    x    (B, T, C, H, W)   image time series, zero padded in time
    doy  (B, T)            day of year
    mask (B, T)            True = real date
and return
    tokens (B, T', N, D)   T' time tokens, N = (H/p) * (W/p) spatial tokens
    doy    (B, T')         mean day of year of the real dates in each time token
    mask   (B, T')         True if the time token contains at least one real date

    patch2d  p x p patch per date             (ViT, TSViT)      T' = T
    tubelet  pt dates x p x p, small blocks   (Prithvi-EO-2.0)  T' = T / pt
    cuboid   pt dates x p x p, large blocks   (Earthformer)     T' = T / pt

Note: tubelet and cuboid use the same operation here and differ only in block size.
Earthformer's "cuboid attention" (attention restricted to within blocks) is an
attention-structure choice; add it in builder.py as a new `attention` option if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _group_time(doy: torch.Tensor, mask: torch.Tensor, pt: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, t = doy.shape
    d = doy.view(b, t // pt, pt).float()
    m = mask.view(b, t // pt, pt)
    count = m.sum(-1)
    mean_doy = (d * m).sum(-1) / count.clamp(min=1)
    return mean_doy.round().long(), count > 0


class Patch2D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)

    def forward(self, x, doy, mask):
        b, t, c, h, w = x.shape
        tok = self.proj(x.reshape(b * t, c, h, w))            # (B*T, D, h', w')
        tok = tok.flatten(2).transpose(1, 2)                  # (B*T, N, D)
        return tok.reshape(b, t, -1, tok.shape[-1]), doy, mask


class Tubelet(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int, pt: int):
        super().__init__()
        self.patch, self.pt = patch, pt
        self.proj = nn.Conv3d(in_ch, dim, (pt, patch, patch), (pt, patch, patch))

    def forward(self, x, doy, mask):
        b, t, c, h, w = x.shape
        if t % self.pt:
            raise ValueError(f"max_dates={t} is not divisible by tubelet_t={self.pt}")
        tok = self.proj(x.permute(0, 2, 1, 3, 4))             # (B, D, T', h', w')
        tok = tok.permute(0, 2, 3, 4, 1)                      # (B, T', h', w', D)
        tok = tok.reshape(b, tok.shape[1], -1, tok.shape[-1])  # (B, T', N, D)
        doy_t, mask_t = _group_time(doy, mask, self.pt)
        return tok, doy_t, mask_t


def build_tokenizer(name: str, in_ch: int, dim: int, patch: int, pt: int) -> nn.Module:
    if name == "patch2d":
        if pt != 1:
            raise ValueError("patch2d uses one date per token: set tubelet_t to 1")
        return Patch2D(in_ch, dim, patch)
    if name in ("tubelet", "cuboid"):
        if pt < 2:
            raise ValueError(f"{name} needs tubelet_t >= 2")
        return Tubelet(in_ch, dim, patch, pt)
    raise ValueError(f"Unknown tokenizer '{name}'")

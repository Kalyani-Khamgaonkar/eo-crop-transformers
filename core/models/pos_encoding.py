"""SO2: positional encoding.

Spatial (where a token sits in the 2D grid)
    learnable   one trained vector per grid position          (ViT, Prithvi)
    sinusoidal  fixed 2D sine / cosine pattern                (TSViT-style, no training)
    relative    learned bias added inside attention, depends  (Shaw et al. 2018; Swin)
                only on the offset between two tokens
    none        no spatial information (ablation)

Temporal (when a token was observed)
    doy_sin        fixed sine / cosine of the day of year      (irregular dates handled)
    doy_learned    trained vector per day of year (1..366)     (TSViT)
    index_learned  trained vector per token index, ignores real dates (plain ViT)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sincos_1d(pos: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """pos (...,) -> (..., dim)"""
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=pos.device) / half)
    ang = pos.float().unsqueeze(-1) * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def sincos_2d(gh: int, gw: int, dim: int) -> torch.Tensor:
    """(gh*gw, dim): half the channels encode the row, half the column."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos"
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.cat([sincos_1d(ys.flatten(), dim // 2), sincos_1d(xs.flatten(), dim // 2)], dim=-1)


class SpatialPE(nn.Module):
    """Additive spatial encoding (learnable / sinusoidal / none). Returns (N, D) or None."""

    def __init__(self, kind: str, gh: int, gw: int, dim: int):
        super().__init__()
        self.kind = kind
        n = gh * gw
        if kind == "learnable":
            self.pe = nn.Parameter(torch.zeros(n, dim))
            nn.init.trunc_normal_(self.pe, std=0.02)
        elif kind == "sinusoidal":
            self.register_buffer("pe", sincos_2d(gh, gw, dim), persistent=False)
        elif kind in ("relative", "none"):
            self.pe = None
        else:
            raise ValueError(f"Unknown spatial_pe '{kind}'")

    def forward(self) -> torch.Tensor | None:
        return self.pe


class RelativeBias2D(nn.Module):
    """Learned attention bias per head for every (dy, dx) offset in a gh x gw grid."""

    def __init__(self, gh: int, gw: int, heads: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros((2 * gh - 1) * (2 * gw - 1), heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        coords = torch.stack([ys.flatten(), xs.flatten()])        # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]             # (2, N, N)
        idx = (rel[0] + gh - 1) * (2 * gw - 1) + (rel[1] + gw - 1)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.table[self.idx].permute(2, 0, 1)              # (heads, N, N)


class TemporalEncoding(nn.Module):
    """Returns (B, T', D) to add to every spatial token of that time step."""

    def __init__(self, kind: str, dim: int, max_tokens: int):
        super().__init__()
        self.kind, self.dim = kind, dim
        if kind == "doy_learned":
            self.emb = nn.Embedding(367, dim)
        elif kind == "index_learned":
            self.emb = nn.Embedding(max_tokens, dim)
        elif kind != "doy_sin":
            raise ValueError(f"Unknown temporal_enc '{kind}'")

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        if self.kind == "doy_sin":
            return sincos_1d(doy, self.dim, max_period=1000.0)
        if self.kind == "doy_learned":
            return self.emb(doy.clamp(0, 366))
        idx = torch.arange(doy.shape[1], device=doy.device).expand_as(doy)
        return self.emb(idx)

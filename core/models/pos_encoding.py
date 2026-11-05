"""SO2: positional encoding.

Spatial (where a token sits in the 2D grid)
    learnable   one trained vector per grid position          (ViT, Prithvi)
    sinusoidal  fixed 2D sine / cosine pattern                (TSViT-style, no training)
    relative    learned bias added inside attention, depends  (Shaw et al. 2018; Swin)
                only on the offset between two tokens
    none        no spatial information (ablation)

Temporal (when a token was observed)
    doy_sin        fixed sine / cosine of the acquisition day  (irregular dates handled)
    doy_learned    trained vector per acquisition day          (TSViT)
    (PASTIS spans ~15 months, so "day" = days since 2018-09-01, range 0..~460)
    index_learned  trained vector per token index, ignores real dates (plain ViT)
    index_sin      fixed sine / cosine of the token index (frame number)  (Prithvi, TiMo)
    none           no temporal information (ablation, "no PE" floor)

Learned tables (doy_learned, index_learned) start at std 0.02, like the learnable spatial PE,
so every learned position table starts from the same small values (nn.Embedding default is std 1).

Relative 3D (pe_mode rel3d): one learned bias per head for every (dt, dy, dx) offset between
two tokens, added to the attention scores (Video Swin / AgriFM). dt is the frame offset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

MAX_DAY = 511  # largest day index the learned temporal table can hold


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
            self.emb = nn.Embedding(MAX_DAY + 1, dim)
        elif kind == "index_learned":
            self.emb = nn.Embedding(max_tokens, dim)
        elif kind not in ("doy_sin", "index_sin", "none"):
            raise ValueError(f"Unknown temporal_enc '{kind}'")
        if hasattr(self, "emb"):
            nn.init.trunc_normal_(self.emb.weight, std=0.02)   # same start as the learnable spatial PE

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(doy.shape[1], device=doy.device).expand_as(doy)
        if self.kind == "doy_sin":
            return sincos_1d(doy, self.dim, max_period=1000.0)
        if self.kind == "index_sin":
            return sincos_1d(idx, self.dim, max_period=1000.0)
        if self.kind == "none":
            return torch.zeros(*doy.shape, self.dim, device=doy.device)
        if self.kind == "doy_learned":
            return self.emb(doy.clamp(0, MAX_DAY))
        return self.emb(idx)


class RelativeBias3D(nn.Module):
    """Learned attention bias per head for every (dt, dy, dx) offset in a t x gh x gw token grid.
    Tokens are ordered time-major, then row, then column (as in the joint encoder)."""

    def __init__(self, t: int, gh: int, gw: int, heads: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros((2 * t - 1) * (2 * gh - 1) * (2 * gw - 1), heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        ts, ys, xs = torch.meshgrid(torch.arange(t), torch.arange(gh), torch.arange(gw), indexing="ij")
        c = torch.stack([ts.flatten(), ys.flatten(), xs.flatten()])   # (3, N)
        rel = c[:, :, None] - c[:, None, :]                            # (3, N, N)
        idx = ((rel[0] + t - 1) * (2 * gh - 1) + (rel[1] + gh - 1)) * (2 * gw - 1) + (rel[2] + gw - 1)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.table[self.idx].permute(2, 0, 1)                   # (heads, N, N)

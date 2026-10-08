"""Build a model from design choices instead of one file per model.

    model = build_model(cfg)   # cfg = resolved config (protocol + overrides)

Design axes (config keys under `model`):
    tokenizer     patch2d | tubelet | cuboid          (SO1)
    attention     temporal_first | joint              (SO1, how time is processed)
    spatial_pe    learnable | sinusoidal | relative | none   (SO2)
    temporal_enc  doy_sin | doy_learned | index_learned

Pipeline
    tokens -> + spatial PE + temporal encoding -> encoder -> pool over time -> pixel head

    joint           one encoder over all T' x N tokens (depth_temporal + depth_spatial blocks)
    temporal_first  temporal encoder per spatial location, masked mean over time,
                    then spatial encoder over N tokens (TSViT order; TSViT itself uses class
                    tokens instead of mean pooling, see core/models/tsvit.py for the reference)

Both structures use the same total number of blocks and the same head,
so parameter counts stay close (checked against the protocol budget).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.models.blocks import Encoder
from core.models.pos_encoding import RelativeBias2D, SpatialPE, TemporalEncoding
from core.models.tokenizers import build_tokenizer


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class PixelHead(nn.Module):
    """Shared by every configuration: token -> p x p pixel logits."""

    def __init__(self, dim: int, num_classes: int, patch: int, gh: int, gw: int):
        super().__init__()
        self.num_classes, self.patch, self.gh, self.gw = num_classes, patch, gh, gw
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes * patch * patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, N, D) -> (B, K, H, W)
        b = x.shape[0]
        p, k = self.patch, self.num_classes
        y = self.fc(self.norm(x)).view(b, self.gh, self.gw, k, p, p)
        return y.permute(0, 3, 1, 4, 2, 5).reshape(b, k, self.gh * p, self.gw * p)


class SITSTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        dim, heads = m["dim"], m["heads"]
        p, pt = m["patch_size"], m["tubelet_t"] if m["tokenizer"] != "patch2d" else 1
        self.gh = self.gw = d["img_size"] // p
        n_tokens_t = d["max_dates"] // pt
        self.attention = m["attention"]

        self.tokenizer = build_tokenizer(m["tokenizer"], d["bands"], dim, p, pt)
        self.spatial_pe = SpatialPE(m["spatial_pe"], self.gh, self.gw, dim)
        self.rel_bias = RelativeBias2D(self.gh, self.gw, heads) if m["spatial_pe"] == "relative" else None
        self.temporal_enc = TemporalEncoding(m["temporal_enc"], dim, n_tokens_t)

        common = dict(heads=heads, mlp_ratio=m["mlp_ratio"], dropout=m["dropout"], drop_path=m["drop_path"])
        if self.attention == "joint":
            self.encoder = Encoder(dim, m["depth_temporal"] + m["depth_spatial"], **common)
        elif self.attention == "temporal_first":
            self.temporal = Encoder(dim, m["depth_temporal"], **common)
            self.spatial = Encoder(dim, m["depth_spatial"], **common)
        else:
            raise ValueError(f"Unknown attention '{self.attention}'")

        self.head = PixelHead(dim, d["num_classes"], p, self.gh, self.gw)
        self.apply(_init_weights)

    def forward(self, img: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok, doy_t, mask_t = self.tokenizer(img, doy, mask)          # (B, T', N, D)
        b, t, n, dim = tok.shape
        spe = self.spatial_pe()
        if spe is not None:
            tok = tok + spe[None, None]
        tok = tok + self.temporal_enc(doy_t)[:, :, None, :]
        bias = self.rel_bias() if self.rel_bias is not None else None  # (h, N, N)

        if self.attention == "joint":
            x = tok.reshape(b, t * n, dim)                            # time-major order
            key_mask = mask_t[:, :, None].expand(b, t, n).reshape(b, t * n)
            full_bias = bias.repeat(1, t, t) if bias is not None else None
            x = self.encoder(x, key_mask, full_bias).view(b, t, n, dim)
            x = masked_mean(x, mask_t[:, :, None].expand(b, t, n), dim=1)   # (B, N, D)
        else:
            x = tok.permute(0, 2, 1, 3).reshape(b * n, t, dim)        # one sequence per location
            key_mask = mask_t[:, None, :].expand(b, n, t).reshape(b * n, t)
            x = self.temporal(x, key_mask)
            x = masked_mean(x, key_mask, dim=1).view(b, n, dim)
            x = self.spatial(x, None, bias)
        return self.head(x)


def _init_weights(mod: nn.Module) -> None:
    if isinstance(mod, nn.Linear):
        nn.init.trunc_normal_(mod.weight, std=0.02)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
    elif isinstance(mod, nn.LayerNorm):
        nn.init.ones_(mod.weight)
        nn.init.zeros_(mod.bias)


def build_model(cfg: dict) -> nn.Module:
    return SITSTransformer(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_param_budget(model: nn.Module, cfg: dict) -> tuple[int, bool, str]:
    n = count_params(model)
    budget, tol = float(cfg["model"]["param_budget"]), float(cfg["model"]["param_tolerance"])
    ok = abs(n - budget) <= tol * budget
    msg = f"{n / 1e6:.3f}M params (budget {budget / 1e6:.2f}M +/- {tol:.0%})"
    return n, ok, msg


def tokens_per_sample(cfg: dict) -> int:
    m, d = cfg["model"], cfg["data"]
    pt = m["tubelet_t"] if m["tokenizer"] != "patch2d" else 1
    return (d["max_dates"] // pt) * (d["img_size"] // m["patch_size"]) ** 2

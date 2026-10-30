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
                    tokens instead of mean pooling)

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

        self.pe_mode = m.get("pe_mode", "add")                      # add | concat | conv
        self.G = 3 if m["tokenizer"] == "groups" else 1
        if m["tokenizer"] == "groups":
            from core.models.pe_variants import SpectralGroupTokenizer
            assert pt == 1, "groups uses one date per token: set tubelet_t to 1"
            self.tokenizer = SpectralGroupTokenizer(d["bands"], dim, p)
        else:
            self.tokenizer = build_tokenizer(m["tokenizer"], d["bands"], dim, p, pt)
        self.spatial_pe = SpatialPE(m["spatial_pe"], self.gh, self.gw, dim)
        self.rel_bias = RelativeBias2D(self.gh, self.gw, heads) if m["spatial_pe"] == "relative" else None
        self.temporal_enc = TemporalEncoding(m["temporal_enc"], dim, n_tokens_t)
        self.group_enc = m.get("group_enc", "learned")              # learned | sin (groups only)
        use_eg = self.G > 1 and not (self.pe_mode == "concat" and self.group_enc == "sin")
        self.group_emb = nn.Parameter(torch.zeros(self.G, dim)) if use_eg else None
        if self.group_emb is not None:
            nn.init.trunc_normal_(self.group_emb, std=0.02)
        if self.pe_mode != "add":
            assert m["spatial_pe"] == "none", "pe_mode concat/conv: set spatial_pe none"
            assert m["temporal_enc"] == "doy_sin", "pe_mode concat/conv: set temporal_enc doy_sin (not used, no parameters)"
        if self.G > 1 and self.rel_bias is not None:
            raise ValueError("relative spatial PE is not supported with spectral groups")

        common = dict(heads=heads, mlp_ratio=m["mlp_ratio"], dropout=m["dropout"], drop_path=m["drop_path"])
        if self.attention == "joint":
            self.encoder = Encoder(dim, m["depth_temporal"] + m["depth_spatial"], **common)
        elif self.attention == "temporal_first":
            self.temporal = Encoder(dim, m["depth_temporal"], **common)
            self.spatial = Encoder(dim, m["depth_spatial"], **common)
        else:
            raise ValueError(f"Unknown attention '{self.attention}'")

        if self.pe_mode == "conv":
            assert self.attention == "joint", "conv PE is implemented for joint attention"
            from core.models.pe_variants import ConvMLP
            for blk in self.encoder.blocks:
                blk.mlp = ConvMLP(dim, m["mlp_ratio"], m["dropout"])
        self.head = PixelHead(dim, d["num_classes"], p, self.gh, self.gw)
        self.apply(_init_weights)

    def forward(self, img: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok, doy_t, mask_t = self.tokenizer(img, doy, mask)          # (B, T', G*N, D)
        b, t, gn, dim = tok.shape
        G = self.G
        n = gn // G
        if self.pe_mode == "add":
            spe = self.spatial_pe()
            if spe is not None:
                tok = tok + spe.repeat(G, 1)[None, None]
            tok = tok + self.temporal_enc(doy_t)[:, :, None, :]
        elif self.pe_mode == "concat":
            from core.models.pe_variants import concat_sincos_pe
            tok = tok + concat_sincos_pe(doy_t, self.gh, self.gw, G, dim, group_sin=(G > 1 and self.group_enc == "sin"))
        if self.group_emb is not None:
            tok = tok + self.group_emb.repeat_interleave(n, dim=0)[None, None]
        bias = self.rel_bias() if self.rel_bias is not None else None  # (h, N, N)

        if self.attention == "joint":
            x = tok.reshape(b, t * gn, dim)                           # time-major, then group, then grid
            key_mask = mask_t[:, :, None].expand(b, t, gn).reshape(b, t * gn)
            full_bias = bias.repeat(1, t, t) if bias is not None else None
            if self.pe_mode == "conv":
                for blk in self.encoder.blocks:
                    blk.mlp.set_grid(t, G, self.gh, self.gw, mask_t)
            x = self.encoder(x, key_mask, full_bias).view(b, t, gn, dim)
            x = masked_mean(x, mask_t[:, :, None].expand(b, t, gn), dim=1)   # (B, G*N, D)
            if G > 1:
                x = x.view(b, G, n, dim).mean(1)                       # merge spectral groups before the head
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


_tokens_per_sample_base = tokens_per_sample


def tokens_per_sample(cfg: dict) -> int:  # noqa: F811  (spectral groups: 3 tokens per date and patch)
    m = cfg["model"]
    if m["tokenizer"] == "groups":
        return 3 * _tokens_per_sample_base({**cfg, "model": {**m, "tokenizer": "patch2d", "tubelet_t": 1}})
    return _tokens_per_sample_base(cfg)

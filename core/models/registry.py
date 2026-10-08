"""Named presets, so you can say `--model vit` like in eo4tsa.

A preset is only a set of design-axis overrides on top of the protocol.
Any preset can still be changed further with --set model.xxx=...
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    # BASELINE ViT: 2D 4x4 patches, joint attention over all tokens,
    # learnable spatial PE, day-of-year temporal encoding (handles irregular dates)
    "vit": {
        "model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1,
        "model.attention": "joint", "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin",
    },
    # TSViT order: per-date patches, temporal attention first, then spatial
    "vit_temporal_first": {
        "model.tokenizer": "patch2d", "model.patch_size": 2, "model.tubelet_t": 1,
        "model.attention": "temporal_first", "model.spatial_pe": "learnable", "model.temporal_enc": "doy_learned",
    },
    # Prithvi-style tubelets
    "vit_tubelet": {
        "model.tokenizer": "tubelet", "model.patch_size": 4, "model.tubelet_t": 3,
        "model.attention": "joint", "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin",
    },
    # Earthformer-style large space-time blocks
    "vit_cuboid": {
        "model.tokenizer": "cuboid", "model.patch_size": 6, "model.tubelet_t": 4,
        "model.attention": "joint", "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin",
    },
}


def list_models() -> list[str]:
    return list(PRESETS)


def preset_overrides(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"Unknown model '{name}'. Options: {', '.join(PRESETS)}")
    return dict(PRESETS[name])

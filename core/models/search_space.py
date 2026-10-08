"""Combination generator with constraint checking (proposal section 4.2.3).

    python -m core.cli list_search_space            # print all valid configs + counts
    python -m core.cli list_search_space --csv out.csv

Edit CANDIDATES to change what is compared. Edit `check` to add constraints.
"""

from __future__ import annotations

import copy
import itertools

from core.models.builder import build_model, check_param_budget, tokens_per_sample

CANDIDATES = {
    # tokenizer: list of (patch_size, tubelet_t)
    "tokenizer": {
        "patch2d": [(2, 1), (3, 1), (4, 1), (6, 1)],
        "tubelet": [(2, 2), (2, 3), (4, 2), (4, 3), (4, 4)],
        "cuboid":  [(6, 3), (6, 4), (6, 5), (8, 2), (8, 3), (12, 2)],
    },
    "attention": ["temporal_first", "joint"],
    "spatial_pe": ["sinusoidal", "learnable", "relative"],
    "temporal_enc": ["doy_sin", "doy_learned"],
}

# Joint attention cost grows with (tokens)^2; above this the run is too slow / too big.
JOINT_TOKEN_BUDGET = 4096


def check(cfg: dict, check_params: bool = True) -> list[str]:
    """Return the reasons a configuration is invalid (empty list = valid)."""
    m, d = cfg["model"], cfg["data"]
    reasons = []
    p, pt = m["patch_size"], m["tubelet_t"]
    if d["img_size"] % p:
        reasons.append(f"img_size {d['img_size']} not divisible by patch {p}")
    if m["tokenizer"] == "patch2d" and pt != 1:
        reasons.append("patch2d needs tubelet_t = 1")
    if m["tokenizer"] != "patch2d":
        if pt < 2:
            reasons.append(f"{m['tokenizer']} needs tubelet_t >= 2")
        if d["max_dates"] % pt:
            reasons.append(f"max_dates {d['max_dates']} not divisible by tubelet_t {pt}")
        if d["max_dates"] // max(pt, 1) < 4:
            reasons.append("fewer than 4 time tokens left")
    if m["spatial_pe"] == "sinusoidal" and m["dim"] % 4:
        reasons.append("sinusoidal 2D PE needs dim divisible by 4")
    if m["attention"] == "joint" and tokens_per_sample(cfg) > JOINT_TOKEN_BUDGET:
        reasons.append(f"joint attention over {tokens_per_sample(cfg)} tokens > budget {JOINT_TOKEN_BUDGET}")
    if not reasons and check_params:
        # Large blocks make the patch embedding and pixel head big (C*pt*p*p*D and D*K*p*p),
        # which can push the model outside the shared parameter budget.
        _, ok, msg = check_param_budget(build_model(cfg), cfg)
        if not ok:
            reasons.append(msg)
    return reasons


def enumerate_configs(base_cfg: dict) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """All combinations -> (valid configs, [(invalid config, reasons)])."""
    valid, invalid = [], []
    tok_options = [(t, ps, pt) for t, sizes in CANDIDATES["tokenizer"].items() for ps, pt in sizes]
    for (tok, ps, pt), att, spe, ten in itertools.product(
        tok_options, CANDIDATES["attention"], CANDIDATES["spatial_pe"], CANDIDATES["temporal_enc"]
    ):
        cfg = copy.deepcopy(base_cfg)
        cfg["model"].update(tokenizer=tok, patch_size=ps, tubelet_t=pt,
                            attention=att, spatial_pe=spe, temporal_enc=ten)
        reasons = check(cfg)
        (invalid.append((cfg, reasons)) if reasons else valid.append(cfg))
    return valid, invalid


def summary_row(cfg: dict) -> dict:
    m = cfg["model"]
    return {
        "tokenizer": m["tokenizer"], "patch_size": m["patch_size"], "tubelet_t": m["tubelet_t"],
        "attention": m["attention"], "spatial_pe": m["spatial_pe"], "temporal_enc": m["temporal_enc"],
        "tokens": tokens_per_sample(cfg),
    }

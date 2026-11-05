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
JOINT_TOKEN_BUDGET_GROUPS = 6480  # spectral groups: 3 tokens per patch and date (option A, agreed 30 Oct)


def check(cfg: dict, check_params: bool = True) -> list[str]:
    """HARD criteria only: the reasons a configuration cannot be built (empty list = valid).

    Agreed 5 Nov: only (1) grid divisibility and (2) positional-encoding shape are hard.
    Token count, parameter budget and memory are soft: see `warnings`; they never reject.
    `check_params` is kept for old callers and has no effect.
    """
    m, d = cfg["model"], cfg["data"]
    reasons = []
    p, pt, D = m["patch_size"], m["tubelet_t"], m["dim"]
    # (1) grid divisibility
    if d["img_size"] % p:
        reasons.append(f"grid: img_size {d['img_size']} not divisible by patch {p}")
    if m["tokenizer"] == "patch2d" and pt != 1:
        reasons.append("grid: patch2d uses one date per token (tubelet_t = 1)")
    if m["tokenizer"] in ("tubelet", "cuboid"):
        if pt < 2:
            reasons.append(f"grid: {m['tokenizer']} needs tubelet_t >= 2")
        elif d["max_dates"] % pt:
            reasons.append(f"grid: max_dates {d['max_dates']} not divisible by tubelet_t {pt}")
    # (2) positional-encoding shape: every position vector must have exactly D numbers
    pe_mode = m.get("pe_mode", "add")
    if m["spatial_pe"] == "sinusoidal" and D % 4:
        reasons.append(f"PE shape: 2D sin-cos needs D divisible by 4 (D = {D})")
    if m["temporal_enc"] in ("doy_sin", "index_sin") and D % 2:
        reasons.append(f"PE shape: sin-cos needs an even D (D = {D})")
    if pe_mode == "concat":
        parts = (D // 4, D // 4, D // 4, D // 4) if (m["tokenizer"] == "groups" and m.get("group_enc") == "sin") \
            else (D // 2, D // 4, D // 4)
        parts = parts[:-1] + (D - sum(parts[:-1]),)
        if sum(parts) != D or any(x % 2 for x in parts):
            reasons.append(f"PE shape: concat split {parts} must be even parts summing to D = {D}")
    return reasons


def soft_warnings(cfg: dict) -> list[str]:
    """SOFT criteria: reported, never a reason to reject (token count, parameter budget, few time tokens)."""
    m, d = cfg["model"], cfg["data"]
    out = []
    n = tokens_per_sample(cfg)
    budget = JOINT_TOKEN_BUDGET_GROUPS if m["tokenizer"] == "groups" else JOINT_TOKEN_BUDGET
    if m["attention"] == "joint" and n > budget:
        out.append(f"tokens: {n} > {budget} (slow / memory heavy, still allowed)")
    if m["tokenizer"] in ("tubelet", "cuboid") and m["tubelet_t"] >= 2 and d["max_dates"] // m["tubelet_t"] < 4:
        out.append("time: fewer than 4 time tokens")
    try:
        _, ok, msg = check_param_budget(build_model(cfg), cfg)
        if not ok:
            out.append(f"params: {msg} (outside budget, still allowed)")
    except Exception as e:  # a build error is reported by the caller
        out.append(f"build: {type(e).__name__}: {e}")
    return out


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

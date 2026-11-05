"""Combination checker: patchification x positional encoding under joint attention.

Level 1 (mathematical): HARD = grid divisibility and PE shape only (agreed 5 Nov).
                        Parameter budget, token count and memory are SOFT: reported, never rejected.
Level 2 (semantic, soft):      flags with the category they belong to; nothing is rejected on these.
Level 3 (cost, measured later): tokens, attention elements, compression, rough time.

Run:  python scripts/check_combinations.py            (prints tables, writes CSV + markdown)
"""
import csv, os

# ---------------------------------------------------------------- fixed setup (final hyperparameters + protocol)
SETUP = dict(H=24, W=24, C=10, T=60, K=19, D=128, depth=8, heads=4, mlp_ratio=4.0, batch=8,
             param_budget=1.7e6, param_tol=0.10, gpu_gb=96, max_gap_days=55, median_gap_days=5,
             days_range=456)  # days since 1 Sep 2018 (0 to 455)
FINAL_HP = "SGD Nesterov, LR 1e-1, WD 5e-4, dropout 0.1, drop path 0.1, flips + date dropout, LS 0, cosine + 10 warmup, batch 8, 100 epochs"
GROUPS = {"RGB+NIR": [0, 1, 2, 6], "red edge": [3, 4, 5, 7], "SWIR": [8, 9]}  # band order B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
BASE_CORE = 1_651_120 - (10*16*128 + 128) - (128*16*19 + 16*19) - 36*128   # baseline transformer without tokenizer/head/spatial PE

# ---------------------------------------------------------------- configurations
# tok: patch2d | tubelet | cuboid | groups ; p: spatial patch ; t: dates per token
# spatial: learnable | sinusoidal | none ; temporal: doy_sin | date_lookup | index_learned | none
# pe: add | concat | conv  (how position enters) ; time_value: date | midpoint | start | none
# eg: group embedding for spectral groups ; code: runnable with current repo?
CONFIGS = [
    dict(id="baseline", tok="patch2d", p=4, t=1, spatial="learnable", temporal="doy_sin", pe="add", time_value="date", code=True),
    dict(id="C1", tok="patch2d", p=4, t=1, spatial="learnable", temporal="date_lookup", pe="add", time_value="date", code=False),
    dict(id="C2", tok="patch2d", p=4, t=1, spatial="none", temporal="none", pe="conv", time_value="none", code=False),
    dict(id="C3", tok="patch2d", p=4, t=1, spatial="sinusoidal", temporal="doy_sin", pe="concat", time_value="date", split=(64, 32, 32), code=False),
    dict(id="C4", tok="tubelet", p=4, t=2, spatial="learnable", temporal="date_lookup", pe="add", time_value="midpoint", code=False),
    dict(id="C5", tok="tubelet", p=4, t=2, spatial="none", temporal="none", pe="conv", time_value="none", code=False),
    dict(id="C6", tok="tubelet", p=4, t=2, spatial="sinusoidal", temporal="doy_sin", pe="concat", time_value="midpoint", split=(64, 32, 32), code=False),
    dict(id="C7", tok="groups", p=4, t=1, spatial="learnable", temporal="date_lookup", pe="add", time_value="date", eg=True, code=False),
    dict(id="C8", tok="groups", p=4, t=1, spatial="none", temporal="none", pe="conv", time_value="none", eg=True, code=False),
    dict(id="C9", tok="groups", p=4, t=1, spatial="sinusoidal", temporal="doy_sin", pe="concat", time_value="date", split=(32, 32, 32, 32), g_sin=True, code=False),
    dict(id="case1", tok="patch2d", p=4, t=1, spatial="sinusoidal", temporal="doy_sin", pe="add", time_value="date", code=True),
    dict(id="case2", tok="patch2d", p=4, t=1, spatial="learnable", temporal="index_learned", pe="add", time_value="index", code=True),
    dict(id="case3", tok="tubelet", p=4, t=2, spatial="learnable", temporal="index_learned", pe="add", time_value="index", code=True),
    dict(id="case4", tok="cuboid", p=4, t=4, spatial="learnable", temporal="doy_sin", pe="add", time_value="start (verify)", code=True),
    # ---- 6 Nov screening (2 patchifications x 6 encoding families; Galileo / OlmoEarth later)
    dict(id="E0_2d", tok="patch2d", p=4, t=1, spatial="none", temporal="none", pe="add", time_value="none", code=True),
    dict(id="E0_tube", tok="tubelet", p=4, t=2, spatial="none", temporal="none", pe="add", time_value="none", code=True),
    dict(id="E1_tube", tok="tubelet", p=4, t=2, spatial="learnable", temporal="doy_sin", pe="add", time_value="midpoint", code=True),
    dict(id="E4_2d", tok="patch2d", p=4, t=1, spatial="sinusoidal", temporal="index_sin", pe="concat", time_value="index", split=(64, 32, 32), code=True),
    dict(id="E4_tube", tok="tubelet", p=4, t=2, spatial="sinusoidal", temporal="index_sin", pe="concat", time_value="index", split=(64, 32, 32), code=True),
    dict(id="E5_2d", tok="patch2d", p=4, t=1, spatial="none", temporal="none", pe="rel3d", time_value="offset", code=True),
    dict(id="E5_tube", tok="tubelet", p=4, t=2, spatial="none", temporal="none", pe="rel3d", time_value="offset", code=True),
]

def analyse(c, S=SETUP):
    p, t, D = c["p"], c["t"], S["D"]
    G = 3 if c["tok"] == "groups" else 1
    gh, gw, Tt = S["H"] // p, S["W"] // p, S["T"] // t
    N = Tt * gh * gw * G
    hard, soft = [], []

    # ---- Level 1: hard checks
    if S["H"] % p or S["W"] % p: hard.append(f"grid: 24 not divisible by p={p}")
    if S["T"] % t: hard.append(f"grid: T={S['T']} not divisible by t={t}")
    if c["pe"] == "concat":
        sp = c.get("split", ())
        if sum(sp) != D: hard.append(f"PE shape: split {sp} sums to {sum(sp)}, not D={D}")
        if any(x % 2 for x in sp): hard.append(f"PE shape: odd part in split {sp} (sin/cos pairs need even)")
        if len(sp) != (4 if G > 1 else 3): hard.append(f"PE shape: split {sp} does not cover t, y, x" + (", g" if G > 1 else ""))
    # parameters
    if c["tok"] == "groups":
        emb = sum(len(b) * p * p * D + D for b in GROUPS.values())
        vals = [len(b) * p * p for b in GROUPS.values()]
    else:
        emb = S["C"] * t * p * p * D + D
        vals = [S["C"] * t * p * p]
    head = D * p * p * S["K"] + p * p * S["K"]
    spe = gh * gw * D if c["spatial"] == "learnable" else 0
    tpe = {"date_lookup": S["days_range"] * D, "index_learned": Tt * D}.get(c["temporal"], 0)
    conv = S["depth"] * (27 * D + D) if c["pe"] == "conv" else 0
    conv += (2 * Tt - 1) * (2 * gh - 1) * (2 * gw - 1) * S["heads"] if c["pe"] == "rel3d" else 0
    eg = 3 * D if c.get("eg") else 0
    params = BASE_CORE + emb + head + spe + tpe + conv + eg
    lo, hi = S["param_budget"] * (1 - S["param_tol"]), S["param_budget"] * (1 + S["param_tol"])
    if not lo <= params <= hi: soft.append("(L1 soft) " + f"params: {params/1e6:.2f} M outside {lo/1e6:.2f} to {hi/1e6:.2f} M")
    # memory: naive attention scores stored for backward (bf16), all layers
    mem_naive = S["batch"] * S["heads"] * S["depth"] * N * N * 2 / 1e9
    if mem_naive > S["gpu_gb"]: soft.append("(L1 soft) " + f"memory: naive attention {mem_naive:.0f} GB > {S['gpu_gb']} GB (needs memory-efficient attention)")

    # ---- Level 2: soft (semantic) flags  (category numbers follow the 12-category list)
    if t > 1 and c["time_value"].startswith("start"):
        soft.append("(1) token covers several dates but gets the start date only")
    if t > 1 and c["time_value"] == "midpoint":
        soft.append("(1) midpoint time: interval length is lost; midpoint may be a day with no acquisition")
    if c["temporal"] == "none" and c["pe"] not in ("conv", "rel3d"):
        soft.append("(2) no temporal position signal")
    if c["pe"] == "conv":
        soft.append("(2)(4) conv PE: neighbour order only, real date gaps invisible")
        soft.append("(mask) conv runs across padded / dropped dates: needs mask handling")
    if c["time_value"] in ("index", "offset"):
        soft.append("(4) frame index: irregular gaps (5 to 55 days) treated as equal steps")
    if t > 1:
        soft.append(f"(3) {t} dates merged per token: span up to ~{(t-1)*S['max_gap_days']} days (median ~{(t-1)*S['median_gap_days']})")
        soft.append("(mask) tubelet may pair a real date with a padded / dropped date")
    ratio = max(vals) / D
    if ratio > 2: soft.append(f"(10) compression {max(vals)} values -> {D} dims ({ratio:.1f}x)")
    if p >= 6: soft.append(f"(11) coarse output: {p}x{p}-pixel blocks blur small parcels")
    if c["temporal"] == "date_lookup":
        soft.append("(6) absolute date lookup: each S2 tile has its own calendar -> date can reveal the tile (shortcut, RQ2.2)")
    if G > 1 and not c.get("eg") and not c.get("g_sin"):
        soft.append("(4*) groups share position at the same (t, y, x): add group embedding E_G")
    if c.get("g_sin"):
        soft.append("(1) sinusoid on group index treats a category as ordered")
    if c["pe"] == "concat" and min(c.get("split", (D,))) < 48:
        soft.append(f"(9) concat split {c.get('split')}: few dims per axis (coarser position resolution)")
    if c["pe"] == "conv" and c["tok"] == "tubelet":
        soft.append("(8) VistaFormer used per-date spatial attention; here joint attention (source mismatch)")
    if N / 2160 > 2:
        soft.append(f"(12) {N/2160:.1f}x tokens of baseline ({(N/2160)**2:.0f}x attention): unequal compute budget")

    # ---- Level 3: cost
    r = N / 2160
    est_min = 9 * (0.5 * r + 0.5 * r * r)   # rough: half linear, half quadratic cost, baseline 9 min / 100 epochs on 5%
    return dict(id=c["id"], tok=c["tok"], patch=f"{t}x{p}x{p}" + (" xG3" if G > 1 else ""), pe=f"{c['spatial']} + {c['temporal']} ({c['pe']})",
                time_value=c["time_value"], tokens=N, attn_M=round(N * N / 1e6, 1), params_M=round(params / 1e6, 3),
                compression=round(ratio, 2), mem_naive_GB=round(mem_naive, 1), est_min_5pct=round(est_min),
                level1="PASS" if not hard else "FAIL", hard="; ".join(hard), n_soft=len(soft), soft=soft,
                runnable_now="yes" if c["code"] else "needs code")

def main():
    rows = [analyse(c) for c in CONFIGS]
    print("FIXED SETUP:", {k: SETUP[k] for k in ("H", "W", "C", "T", "D", "depth", "heads", "batch")})
    print("FINAL HYPERPARAMETERS:", FINAL_HP, "\n")
    hdr = f"{'id':9} {'patch':11} {'tokens':>6} {'attn M':>7} {'params M':>8} {'compr':>5} {'naive GB':>8} {'~min':>5} {'L1':5} {'soft':>4}  runnable"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['id']:9} {r['patch']:11} {r['tokens']:6d} {r['attn_M']:7.1f} {r['params_M']:8.3f} {r['compression']:5.2f} "
              f"{r['mem_naive_GB']:8.1f} {r['est_min_5pct']:5d} {r['level1']:5} {r['n_soft']:4d}  {r['runnable_now']}")
    print("\nLEVEL 1 FAILS + LEVEL 2 FLAGS")
    for r in rows:
        print(f"\n{r['id']}  [{r['patch']}, {r['pe']}, time = {r['time_value']}]")
        if r["hard"]: print("  HARD:", r["hard"])
        for s in r["soft"] or ["no semantic flags"]: print("  -", s)
    out = os.path.expanduser("~/outputs/reports"); os.makedirs(out, exist_ok=True)
    with open(f"{out}/combination_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "soft"] + ["soft_flags"]); w.writeheader()
        for r in rows: w.writerow({**{k: v for k, v in r.items() if k != "soft"}, "soft_flags": " | ".join(r["soft"])})
    with open(f"{out}/combination_check.md", "w") as f:
        f.write(f"# Combination check\n\nFixed setup: 24x24 crops, 10 bands, T=60 (padded), D=128, 8 blocks, 4 heads, batch 8.\n\n"
                f"Final hyperparameters: {FINAL_HP}.\n\n| id | patch | PE | tokens | attn (M) | params (M) | compression | Level 1 | semantic flags | runnable |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['patch']} | {r['pe']} | {r['tokens']} | {r['attn_M']} | {r['params_M']} | {r['compression']} | "
                    f"{r['level1']}{' ('+r['hard']+')' if r['hard'] else ''} | {'<br>'.join(r['soft']) or 'none'} | {r['runnable_now']} |\n")
    print(f"\nsaved {out}/combination_check.csv and combination_check.md")

if __name__ == "__main__":
    main()

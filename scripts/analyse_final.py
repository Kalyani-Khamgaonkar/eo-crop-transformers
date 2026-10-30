import re, glob, yaml, statistics as st
runs = yaml.safe_load(open("experiments/hp_final.yaml"))["runs"]
EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)\s+(\d+)s")
def read(path):
    try: t = open(path).read()
    except FileNotFoundError: return None
    ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}
    m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
    if not m: return dict(status=f"running, epoch {max(ep) if ep else 0}")
    b = int(m.group(2)); r = ep.get(b)
    return dict(val=float(m.group(1)), best=b, gap=r[2]-r[3] if r else None, oa=r[5] if r else None,
                sec=st.median(v[6] for v in ep.values()) if ep else None)
R = {r["id"]: read(f"logs/run_hp_final_{i}.out") for i, r in enumerate(runs)}
S2 = lambda i: read(f"logs/run_hp_stage2_{i}.out")
f = lambda d: d.get("status") if d and "status" in d else ("not started" if d is None else
    f"val {d['val']:.3f}  gap {d['gap']:.3f}  OA {d['oa']:.3f}  best ep {d['best']}  {d['sec']:.0f} s/epoch")
print("=== FOLD I, FULL DATA (Stage 2 winner) ===")
print("  " + f(R["fold1_full"]))
print("\n=== STAGE 3: scheduler x warmup (winner config, seed 42) ===")
print(f"  {'cosine_wu10 (Stage 2 winner)':30} " + f(S2(95)))
for k in sorted(k for k in R if k.startswith("s3_")): print(f"  {k[3:]:30} " + f(R[k]))
print("\n=== STAGE 4: seeds (seed 42 from Stage 2) ===")
for k, run42 in [("top1", 95), ("top2", 56), ("top3", 44)]:
    ds = [S2(run42)] + [R.get(f"s4_{k}_seed{s}") for s in (1, 2)]
    vals = [d["val"] for d in ds if d and "val" in d]
    gaps = [d["gap"] for d in ds if d and d.get("gap") is not None]
    sd = f" +/- {st.stdev(vals):.3f}" if len(vals) > 1 else ""
    print(f"  {k}: seeds 42/1/2 = " + ", ".join(f"{v:.3f}" for v in vals) + f"  -> mean {st.mean(vals):.3f}{sd}  mean gap {st.mean(gaps):.3f}  (n={len(vals)})")

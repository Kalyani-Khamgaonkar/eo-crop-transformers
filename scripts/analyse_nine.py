import re, yaml, statistics as st
EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)\s+(\d+)s")
res = {}
for exp in ["nine_c1_c4", "nine_rest"]:
    for i, r in enumerate(yaml.safe_load(open(f"experiments/{exp}.yaml"))["runs"]):
        try: t = open(f"logs/run_{exp}_{i}.out").read()
        except FileNotFoundError: continue
        m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
        if not m: continue
        ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}; b = int(m.group(2))
        cid = r["id"][:2]
        res.setdefault(cid, []).append(dict(val=float(m.group(1)), gap=ep[b][2]-ep[b][3] if b in ep else float("nan"),
                                            oa=ep[b][5] if b in ep else float("nan"), sec=st.median(v[6] for v in ep.values())))
cell = lambda c: (lambda v: f"{st.mean(v):.3f}" + (f" +/- {st.stdev(v):.3f}" if len(v) > 1 else "") + f" (n={len(v)})")([d["val"] for d in res[c]]) if c in res else "-"
print(f"{'val mIoU':16} {'Learnable date lookup':24} {'Conv PE':24} {'Sinusoidal concat':24}")
for name, ids in [("2D", ["C1", "C2", "C3"]), ("Tubelet 2", ["C4", "C5", "C6"]), ("Spectral groups", ["C7", "C8", "C9"])]:
    print(f"{name:16} " + " ".join(f"{cell(c):24}" for c in ids))
print("\nid   n  val mIoU         gap    OA     s/epoch")
for c in sorted(res):
    v = [d["val"] for d in res[c]]
    print(f"{c}   {len(v)}  {st.mean(v):.3f} +/- {st.stdev(v) if len(v) > 1 else 0:.3f}  {st.mean(d['gap'] for d in res[c]):.3f}  "
          f"{st.mean(d['oa'] for d in res[c]):.3f}  {st.mean(d['sec'] for d in res[c]):.0f}")
print("\nReference: baseline (2D, learnable spatial + date sinusoid) 0.454 +/- 0.015 (3 seeds)")

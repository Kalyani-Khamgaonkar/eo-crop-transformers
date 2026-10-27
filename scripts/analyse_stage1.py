import re, yaml, glob, csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = yaml.safe_load(open("experiments/hp_stage1.yaml"))["runs"]
EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)")
LRS = {"sgd": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}; LRA = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
OPTS = ["adamw", "adam", "nadam", "radam", "rmsprop", "sgd"]
res, curves = [], {}
for f in glob.glob("logs/run_hp_stage1_*.out"):
    i = int(re.search(r"_(\d+)\.out", f).group(1)); t = open(f).read()
    m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
    if not m: continue
    ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}; curves[i] = ep
    b = int(m.group(2)); r = runs[i]; row = ep.get(b)
    res.append(dict(run=i, opt=r["train.optimizer"], lr=float(r["train.lr"]), wd=float(r["train.weight_decay"]),
        val=float(m.group(1)), train=row[2] if row else None, gap=round(row[2]-row[3], 3) if row else None,
        oa=row[5] if row else None, best_ep=b, minloss_ep=min(ep, key=lambda k: ep[k][1]) if ep else None))
res.sort(key=lambda d: -d["val"]); print(f"finished runs: {len(res)} / 168\n")
out = os.path.expanduser("~/outputs/reports"); os.makedirs(out, exist_ok=True)
with open(f"{out}/stage1_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
print("TOP 15"); print(f"{'#':>2} {'run':>3} {'opt':8} {'lr':>7} {'wd':>7} {'val':>6} {'train':>6} {'gap':>6} {'OA':>6} {'best':>4} {'minL':>4}")
for k, d in enumerate(res[:15], 1):
    print(f"{k:2d} {d['run']:3d} {d['opt']:8} {d['lr']:7.0e} {d['wd']:7.0e} {d['val']:6.3f} {f3(d['train']):>6} {f3(d['gap']):>6} {f3(d['oa']):>6} {d['best_ep']:4d} {str(d['minloss_ep']):>4}")
print("\nBEST PER OPTIMISER + EDGE CHECK")
for o in OPTS:
    ds = [d for d in res if d["opt"] == o]; b = ds[0]; g = LRS.get(o, LRA)
    pos = g.index(min(g, key=lambda x: abs(x - b["lr"])))
    print(f"  {o:8} {b['val']:.3f}  lr {b['lr']:.0e}  wd {b['wd']:.0e}  gap {f3(b['gap'])}  {'AT EDGE -> extend' if pos in (0, 6) else 'inside range -> OK'}")
print("\nVAL mIoU (rows = LR, columns = WD)")
for o in OPTS:
    ds = {(d["lr"], d["wd"]): d["val"] for d in res if d["opt"] == o}; wds = sorted({k[1] for k in ds})
    print(f"\n  {o:7} " + "  ".join(f"{w:>7.0e}" for w in wds))
    for lr in LRS.get(o, LRA):
        print(f"  {lr:7.0e} " + "  ".join(f"{ds[(lr, w)]:7.3f}" if (lr, w) in ds else "      -" for w in wds))
top = res[:3]; fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for d in top:
    ep = curves[d["run"]]; xs = sorted(ep); lab = f"{d['opt']} lr {d['lr']:.0e} wd {d['wd']:.0e}"
    l, = ax[0].plot(xs, [ep[x][3] for x in xs], label=lab + " (val)")
    ax[0].plot(xs, [ep[x][2] for x in xs], ls="--", color=l.get_color(), alpha=.6, label=lab + " (train)")
    ax[1].plot(xs, [ep[x][1] for x in xs], color=l.get_color()); ax[1].plot(xs, [ep[x][0] for x in xs], ls="--", color=l.get_color(), alpha=.6)
ax[0].set(title="mIoU per epoch (solid = val, dashed = train)", xlabel="epoch"); ax[1].set(title="Loss per epoch (solid = val, dashed = train)", xlabel="epoch")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{out}/stage1_top3_curves.png", dpi=150)
print(f"\nsaved {out}/stage1_results.csv and stage1_top3_curves.png")

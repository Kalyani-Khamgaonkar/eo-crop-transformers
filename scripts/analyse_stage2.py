import re, yaml, glob, csv, os
from statistics import mean
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = yaml.safe_load(open("experiments/hp_stage2.yaml"))["runs"]
EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)")
res, curves = [], {}
for f in glob.glob("logs/run_hp_stage2_[0-9]*.out"):
    i = int(re.search(r"_(\d+)\.out$", f).group(1)); t = open(f).read()
    m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
    if not m: continue
    ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}; curves[i] = ep
    b = int(m.group(2)); r = runs[i]; row = ep.get(b)
    res.append(dict(run=i, opt=r["train.optimizer"], dropout=float(r["model.dropout"]), drop_path=float(r["model.drop_path"]),
        augment=str(r["train.augment"]).replace("False", "off"), ls=float(r["train.label_smoothing"]),
        val=float(m.group(1)), train=row[2] if row else None, gap=round(row[2] - row[3], 3) if row else None,
        oa=row[5] if row else None, best_ep=b, minloss_ep=min(ep, key=lambda k: ep[k][1]) if ep else None))
res.sort(key=lambda d: -d["val"]); print(f"finished runs: {len(res)} / 108\n")
out = os.path.expanduser("~/outputs/reports"); os.makedirs(out, exist_ok=True)
with open(f"{out}/stage2_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"

print("REPRODUCIBILITY CHECK (Stage 1 settings: dropout 0.1, drop path 0.1, augment off, LS 0)")
for d in res:
    if d["dropout"] == 0.1 and d["drop_path"] == 0.1 and d["augment"] == "off" and d["ls"] == 0:
        ref = {"sgd": 0.433, "adamw": 0.435}[d["opt"]]
        print(f"  {d['opt']:6} Stage 2 {d['val']:.3f}  vs Stage 1 {ref:.3f}  diff {d['val'] - ref:+.3f}")

print("\nTOP 15")
print(f"{'#':>2} {'run':>3} {'opt':6} {'drop':>4} {'dpath':>5} {'augment':10} {'LS':>4} {'val':>6} {'train':>6} {'gap':>6} {'OA':>6} {'best':>4} {'minL':>4}")
for k, d in enumerate(res[:15], 1):
    print(f"{k:2d} {d['run']:3d} {d['opt']:6} {d['dropout']:4.1f} {d['drop_path']:5.1f} {d['augment']:10} {d['ls']:4.1f} "
          f"{d['val']:6.3f} {f3(d['train']):>6} {f3(d['gap']):>6} {f3(d['oa']):>6} {d['best_ep']:4d} {str(d['minloss_ep']):>4}")

print("\nBEST PER OPTIMISER")
for o in ["sgd", "adamw"]:
    ds = [d for d in res if d["opt"] == o]
    if ds:
        b = ds[0]
        print(f"  {o:6} {b['val']:.3f}  dropout {b['dropout']}  drop path {b['drop_path']}  augment {b['augment']}  LS {b['ls']}  gap {f3(b['gap'])}")

print("\nAVERAGE EFFECT OF EACH SETTING (mean val mIoU / mean gap, n = runs; unequal n while runs are missing)")
for key, vals in [("dropout", [0.0, 0.1, 0.2]), ("drop_path", [0.0, 0.1, 0.2]),
                  ("augment", ["off", "flip", "flip_date"]), ("ls", [0.0, 0.1])]:
    print(f"\n  {key}")
    for o in ["sgd", "adamw"]:
        cells = []
        for v in vals:
            ds = [d for d in res if d["opt"] == o and d[key] == v]
            g = [d["gap"] for d in ds if d["gap"] is not None]
            cells.append(f"{v!s:>9}: {mean(d['val'] for d in ds):.3f} / {mean(g):.3f} (n={len(ds):2d})" if ds else f"{v!s:>9}: -")
        print(f"    {o:6} " + "   ".join(cells))

top = res[:3]; fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for d in top:
    ep = curves[d["run"]]; xs = sorted(ep)
    lab = f"{d['opt']} do {d['dropout']} dp {d['drop_path']} {d['augment']} ls {d['ls']}"
    l, = ax[0].plot(xs, [ep[x][3] for x in xs], label=lab + " (val)")
    ax[0].plot(xs, [ep[x][2] for x in xs], ls="--", color=l.get_color(), alpha=.6, label=lab + " (train)")
    ax[1].plot(xs, [ep[x][1] for x in xs], color=l.get_color()); ax[1].plot(xs, [ep[x][0] for x in xs], ls="--", color=l.get_color(), alpha=.6)
ax[0].set(title="mIoU per epoch (solid = val, dashed = train)", xlabel="epoch"); ax[1].set(title="Loss per epoch (solid = val, dashed = train)", xlabel="epoch")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{out}/stage2_top3_curves.png", dpi=150)
print(f"\nsaved {out}/stage2_results.csv and stage2_top3_curves.png")

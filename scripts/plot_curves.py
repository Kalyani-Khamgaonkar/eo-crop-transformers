"""Usage: python scripts/plot_curves.py out.png "label=logfile" ["label=logfile" ...]"""
import re, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)")
out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for label, path in items:
    t = open(path).read(); ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}
    if not ep: print("no epochs in", path); continue
    xs = sorted(ep); m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
    for k, (tr, va, name) in enumerate([(0, 1, "Loss"), (2, 3, "mIoU (macro)"), (4, 5, "OA")]):
        l, = ax[k].plot(xs, [ep[x][va] for x in xs], label=f"{label} val")
        ax[k].plot(xs, [ep[x][tr] for x in xs], ls="--", alpha=.6, color=l.get_color(), label=f"{label} train")
        if m and k == 1:
            b = int(m.group(2)); ax[k].scatter([b], [float(m.group(1))], color=l.get_color(), zorder=5)
        ax[k].set(title=name + " per epoch (solid = val, dashed = train)", xlabel="epoch"); ax[k].grid(alpha=.3)
    print(f"{label:25} epochs {len(xs):3d}  best val mIoU {m.group(1) if m else 'n/a'} at epoch {m.group(2) if m else '-'}")
ax[1].legend(fontsize=7); fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out, dpi=150); print("saved", out)

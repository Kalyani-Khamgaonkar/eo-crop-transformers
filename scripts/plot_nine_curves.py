"""Train and validation curves for the 9 screening models (+ the 2D date-sinusoid reference).

    python scripts/plot_nine_curves.py

Reads logs/run_<experiment>_<i>.out. One panel per model in the same grid as the slides
(rows: 2D patch, tubelet; columns: no position, date sinusoid, learned date table, frame number,
relative distance). Solid = validation, dashed = train, line = mean over seeds, band = min to max.
Writes OUTPUT_DIR/reports/curves_nine_miou.png and curves_nine_loss.png.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core.config import get_paths

EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)")
COLS = ["No position", "Date sinusoid", "Learned date table", "Frame number", "Relative distance"]
ROWS = ["2D patch", "Tubelet"]
GRID = {  # (row, col) -> (experiment, run ids without seed)
    (0, 0): ("pe_screen_6nov", "E0_2d"), (0, 1): ("hp_final", "s4_top2"), (0, 2): ("pe_screen_6nov", "C1b_2d_E2"),
    (0, 3): ("pe_screen_6nov", "E4_2d"), (0, 4): ("pe_screen_6nov", "E5_2d"),
    (1, 0): ("pe_screen_6nov", "E0_tube"), (1, 1): ("pe_screen_6nov", "E1_tube"), (1, 2): ("pe_screen_6nov", "C4b_tube_E2"),
    (1, 3): ("pe_screen_6nov", "E4_tube"), (1, 4): ("pe_screen_6nov", "E5_tube"),
}
NUM = {(0, 0): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9}


def curves(exp, prefix):
    try:
        runs = yaml.safe_load(open(f"experiments/{exp}.yaml"))["runs"]
    except FileNotFoundError:
        return []
    out = []
    for i, r in enumerate(runs):
        if not r["id"].startswith(prefix):
            continue
        try:
            ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(open(f"logs/run_{exp}_{i}.out").read())}
        except FileNotFoundError:
            continue
        if ep:
            xs = sorted(ep)
            out.append(np.array([ep[x] for x in xs]))   # (epochs, 6): loss tr, val, miou tr, val, oa tr, val
    return out


def plot(metric, tr_col, va_col, ylabel, fname, ylim):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5), sharex=True, sharey=True)
    for (r, c), (exp, prefix) in GRID.items():
        ax = axes[r, c]
        cs = curves(exp, prefix)
        if not cs:
            ax.text(0.5, 0.5, "no logs", ha="center", transform=ax.transAxes); continue
        n = min(len(x) for x in cs)
        arr = np.stack([x[:n] for x in cs])               # (seeds, epochs, 6)
        xs = np.arange(n)
        for col, ls, color, lab in ((va_col, "-", "#2B6CB0" if r == 0 else "#C2410C", "val"),
                                    (tr_col, "--", "#5A6577", "train")):
            mean = arr[:, :, col].mean(0)
            ax.plot(xs, mean, ls=ls, color=color, lw=2, label=lab)
            ax.fill_between(xs, arr[:, :, col].min(0), arr[:, :, col].max(0), color=color, alpha=0.15, lw=0)
        best = arr[:, :, va_col].max(1).mean() if metric == "miou" else arr[:, :, va_col].min(1).mean()
        tag = "reference" if (r, c) == (0, 1) else f"{NUM[(r, c)]}"
        ax.set_title(f"{tag} · {ROWS[r]} + {COLS[c]}\nbest val {best:.3f}  (n={len(cs)} seeds)", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(*ylim)
        if r == 1:
            ax.set_xlabel("epoch")
        if c == 0:
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc="lower right" if metric == "miou" else "upper right")
    fig.suptitle(f"{ylabel}: solid = validation, dashed = train (mean over seeds, band = min to max), 5% PASTIS fold I",
                 fontsize=13)
    fig.tight_layout()
    out = get_paths().output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, dpi=150)
    print("saved", out / fname)


if __name__ == "__main__":
    plot("miou", 2, 3, "Macro mIoU", "curves_nine_miou.png", (0.0, 0.85))
    plot("loss", 0, 1, "Loss", "curves_nine_loss.png", (0.0, 3.0))

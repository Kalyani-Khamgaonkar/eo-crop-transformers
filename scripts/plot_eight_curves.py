"""Train and validation curves for the 8 main models (same learned P[i]), grid as on the scores slide."""
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
COLS = ["Date sinusoid", "Learned date table", "Frame number, fixed", "Frame number, learned"]
ROWS = ["2D patch", "Tubelet"]
GRID = {
    (0, 0): ("hp_final", "s4_top2"), (0, 1): ("pe_screen_6nov", "C1b_2d_E2"),
    (0, 2): ("frame_learnedP_6nov", "frame_sin_2d"), (0, 3): ("frame_learnedP_6nov", "frame_learned_2d"),
    (1, 0): ("pe_screen_6nov", "E1_tube"), (1, 1): ("pe_screen_6nov", "C4b_tube_E2"),
    (1, 2): ("frame_learnedP_6nov", "frame_sin_tube"), (1, 3): ("frame_learnedP_6nov", "frame_learned_tube"),
}


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
            out.append(np.array([ep[x] for x in sorted(ep)]))
    return out


def plot(metric, tr_col, va_col, ylabel, fname, ylim):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex=True, sharey=True)
    for (r, c), (exp, prefix) in GRID.items():
        ax = axes[r, c]
        cs = curves(exp, prefix)
        if not cs:
            ax.text(0.5, 0.5, "no logs", ha="center", transform=ax.transAxes); continue
        n = min(len(x) for x in cs)
        arr = np.stack([x[:n] for x in cs])
        xs = np.arange(n)
        for col, ls, color, lab in ((va_col, "-", "#2B6CB0" if r == 0 else "#C2410C", "val"), (tr_col, "--", "#5A6577", "train")):
            ax.plot(xs, arr[:, :, col].mean(0), ls=ls, color=color, lw=2, label=lab)
            ax.fill_between(xs, arr[:, :, col].min(0), arr[:, :, col].max(0), color=color, alpha=0.15, lw=0)
        best = arr[:, :, va_col].max(1).mean() if metric == "miou" else arr[:, :, va_col].min(1).mean()
        ax.set_title(f"{r * 4 + c + 1} · {ROWS[r]} + {COLS[c]}\nbest val {best:.3f}  (n={len(cs)} seeds)", fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_ylim(*ylim)
        if r == 1:
            ax.set_xlabel("epoch")
        if c == 0:
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc="lower right" if metric == "miou" else "upper right")
    fig.suptitle(f"{ylabel}, 8 main models (same learned P[i]): solid = validation, dashed = train "
                 f"(mean over seeds, band = min to max), 5% PASTIS fold I", fontsize=13)
    fig.tight_layout()
    out = get_paths().output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, dpi=150)
    print("saved", out / fname)


if __name__ == "__main__":
    plot("miou", 2, 3, "Macro mIoU", "curves_eight_miou.png", (0.0, 0.85))
    plot("loss", 0, 1, "Loss", "curves_eight_loss.png", (0.0, 3.0))

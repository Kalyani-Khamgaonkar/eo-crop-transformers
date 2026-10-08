"""Test evaluation: run ONCE per final configuration, on the full test fold."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.config import get_paths
from core.data.pastis import PASTIS24, split_file
from core.evaluation.metrics import PASTIS_CLASSES
from core.models.builder import build_model
from core.training.train import evaluate
from core.utils.results import append_result


def evaluate_test(run_name: str) -> dict:
    paths = get_paths()
    run_dir = paths.checkpoints / run_name
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])

    d = cfg["data"]
    root = paths.dataset_dir
    ds = PASTIS24(root, split_file(root, d["fold_run"], "test"), max_dates=d["max_dates"],
                  ignore_index=d["ignore_index"], norm_stats=root / "stats" / f"norm_{d['fold_run']}.json")
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=d["num_workers"])
    m = evaluate(model, dl, None, device, cfg)

    out = paths.eval / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps({"best_epoch": state["epoch"], **m}, indent=2))
    with open(out / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "iou", "recall", "precision", "f1", "support"])
        pc = m["per_class"]
        for i in range(len(pc["iou"])):
            name = PASTIS_CLASSES[i] if i < len(PASTIS_CLASSES) else str(i)
            w.writerow([i, name, pc["iou"][i], pc["recall"][i], pc["precision"][i], pc["f1"][i], pc["support"][i]])
    _plot_confusion(m["confusion_matrix"], out / "confusion_matrix.png")

    append_result(paths.output_dir / "results" / "test_runs.csv",
                  {"run_name": run_name, "test_miou": round(m["miou_macro"], 4), "test_oa": round(m["oa"], 4)})
    print(f"[test] {run_name}: mIoU {m['miou_macro']:.4f}  OA {m['oa']:.4f}  -> {out}")
    return m


def _plot_confusion(cm, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    fig.colorbar(im, ax=ax, label="Share of true pixels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

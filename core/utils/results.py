"""One results table for all runs: outputs/results/runs.csv.

Each finished run appends one row. This file becomes the results table in the thesis.
"""

from __future__ import annotations

import csv
import fcntl
from pathlib import Path

COLUMNS = [
    "run_name", "experiment", "run_id", "seed", "fold_run", "subset_fraction",
    "tokenizer", "patch_size", "tubelet_t", "attention", "spatial_pe", "temporal_enc",
    "optimizer", "lr", "weight_decay", "loss", "dropout", "drop_path", "augment", "epochs",
    "best_epoch", "val_miou", "val_oa", "train_oa", "train_val_gap",
    "test_miou", "test_oa",
    "params", "tokens_per_sample", "peak_mem_gb", "sec_per_epoch", "total_hours",
    "git_commit", "slurm_job_id",
]


def append_result(csv_path: Path, row: dict) -> None:
    """Append a row; safe when several Slurm jobs finish at the same time."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def row_from_config(cfg: dict) -> dict:
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    return {
        "run_name": cfg["run"]["name"],
        "experiment": cfg["run"]["experiment"],
        "run_id": cfg["run"]["run_id"],
        "seed": t["seed"],
        "fold_run": d["fold_run"],
        "subset_fraction": d["subset_fraction"],
        "tokenizer": m["tokenizer"],
        "patch_size": m["patch_size"],
        "tubelet_t": m["tubelet_t"],
        "attention": m["attention"],
        "spatial_pe": m["spatial_pe"],
        "temporal_enc": m["temporal_enc"],
        "optimizer": t["optimizer"],
        "lr": t["lr"],
        "weight_decay": t["weight_decay"],
        "loss": t["loss"],
        "dropout": m["dropout"],
        "drop_path": m["drop_path"],
        "augment": t["augment"],
        "epochs": t["epochs"],
    }

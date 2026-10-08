"""Aggregate runs.csv into summary.csv + report.md (mean +/- std over seeds)."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from core.config import get_paths

METRICS = ["val_miou", "val_oa", "train_val_gap", "peak_mem_gb", "sec_per_epoch"]
KEEP = ["experiment", "run_id", "tokenizer", "patch_size", "tubelet_t", "attention",
        "spatial_pe", "temporal_enc", "params", "tokens_per_sample"]


def _num(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_report(experiment: str | None = None, out_dir: Path | None = None) -> Path:
    paths = get_paths()
    out_dir = out_dir or paths.reports
    rows = list(csv.DictReader(open(paths.results_csv))) if paths.results_csv.exists() else []
    if experiment:
        rows = [r for r in rows if r["experiment"] == experiment]
    if not rows:
        raise FileNotFoundError("No finished runs found in runs.csv")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["run_id"])].append(r)

    summary = []
    for (exp, rid), rs in groups.items():
        out = {k: rs[0].get(k, "") for k in KEEP}
        out["n_seeds"] = len(rs)
        for m in METRICS:
            vals = [v for v in (_num(r.get(m)) for r in rs) if v is not None]
            out[f"{m}_mean"] = round(statistics.mean(vals), 4) if vals else ""
            out[f"{m}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else ""
        summary.append(out)
    summary.sort(key=lambda r: -(r["val_miou_mean"] or 0))

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = experiment or "all"
    csv_path = out_dir / f"summary_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    lines = [f"# Results: {tag}", "", "Sorted by validation mIoU (mean over seeds).", "",
             "| run | tokenizer | attention | spatial PE | temporal | val mIoU | val OA | gap | mem GB | s/epoch | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        def pm(m):
            mean, std = r[f"{m}_mean"], r[f"{m}_std"]
            return f"{mean}" + (f" ± {std}" if std != "" else "")
        lines.append(f"| {r['run_id']} | {r['tokenizer']} p{r['patch_size']} t{r['tubelet_t']} | {r['attention']} | "
                     f"{r['spatial_pe']} | {r['temporal_enc']} | {pm('val_miou')} | {pm('val_oa')} | "
                     f"{pm('train_val_gap')} | {pm('peak_mem_gb')} | {pm('sec_per_epoch')} | {r['n_seeds']} |")
    md_path = out_dir / f"report_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[report] {csv_path}\n[report] {md_path}")
    return md_path

#!/bin/bash
# Evaluate every finished 5-fold run on its own test fold, then summarise mean +/- std.
#SBATCH --job-name=eo-eval
#SBATCH --partition=itc-gpu,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh
OUT=${OUTPUT_DIR:-$HOME/outputs}
for d in "$OUT"/checkpoints/baseline_vit_5fold__frozen__*; do
  [ -f "$d/best.pt" ] && python -m core.cli evaluate_test "$(basename "$d")" || echo "[eval] skipped $d"
done
python -m core.cli report --experiment baseline_vit_5fold
python - "$OUT" <<'PY'
import csv, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open(out / "results" / "test_runs.csv")) if r["run_name"].startswith("baseline_vit_5fold")]
latest = {r["run_name"]: r for r in rows}
miou = [float(r["test_miou"]) for r in latest.values()]
oa = [float(r["test_oa"]) for r in latest.values()]
lines = ["# Baseline ViT, full data, 5-fold TEST results", ""]
lines += [f"- {k}: test mIoU {float(v['test_miou']):.4f}, OA {float(v['test_oa']):.4f}" for k, v in sorted(latest.items())]
if miou:
    sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0
    lines += ["", f"**Test mIoU {statistics.mean(miou):.4f} +/- {sd(miou):.4f}, OA {statistics.mean(oa):.4f} +/- {sd(oa):.4f} over {len(miou)} folds**"]
(out / "reports").mkdir(exist_ok=True)
(out / "reports" / "test_baseline_vit_5fold.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

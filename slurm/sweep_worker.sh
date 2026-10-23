#!/bin/bash
# Run a whole experiment file with W GPU workers (stays under the per-user job submit limit).
#   sbatch --array=0-3 --export=ALL,W=4 slurm/sweep_worker.sh experiments/hp_stage1.yaml
# Worker k runs indices k, k+W, k+2W, ... one after another.
#SBATCH --job-name=eo-worker
#SBATCH --partition=itc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh
CONFIG=$1; shift
W=${W:-4}
N=$(python -m core.cli num_runs "$CONFIG" | tail -n 1)
for ((i=SLURM_ARRAY_TASK_ID; i<N; i+=W)); do
  echo "[worker] run $i of $N"
  python -m core.cli train --experiment-config "$CONFIG" --run-index "$i" "$@" || echo "[worker] run $i FAILED"
done
echo "[worker] finished"

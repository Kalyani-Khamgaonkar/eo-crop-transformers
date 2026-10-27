#!/bin/bash
# P runs at the same time on one GPU; any number of jobs share one ordered queue.
#   sbatch --array=0-1 --export=ALL,P=4 slurm/sweep_packed.sh experiments/hp_stage1.yaml
# Each slot claims the next unclaimed run (atomic mkdir), so runs go in file order.
#SBATCH --job-name=eo-packed
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
P=${P:-4}
N=$(python -m core.cli num_runs "$CONFIG" | tail -n 1)
CL="claims/$(basename "$CONFIG" .yaml)"; mkdir -p "$CL"
if [ "${CLEAN:-0}" = 1 ]; then for d in "$CL"/*; do [ -d "$d" ] || continue; i=$(basename "$d"); grep -q "\[done\]" "logs/run_$(basename "$CONFIG" .yaml)_${i}.out" 2>/dev/null || rmdir "$d"; done; fi
for ((p=0; p<P; p++)); do
  (
    for ((i=0; i<N; i++)); do
      mkdir "$CL/$i" 2>/dev/null || continue
      echo "[slot $p] run $i of $N"
      python -m core.cli train --experiment-config "$CONFIG" --run-index "$i" --set data.num_workers=2 "$@" \
        > "logs/run_$(basename "$CONFIG" .yaml)_${i}.out" 2>&1 \
        && echo "[slot $p] run $i done" || echo "[slot $p] run $i FAILED"
    done
  ) &
done
wait
echo "[packed] finished"

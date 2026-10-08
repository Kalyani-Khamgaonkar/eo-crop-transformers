#!/bin/bash
# Run every entry of an experiment file as one Slurm job array (one GPU per run).
#   N=$(python -m core.cli num_runs experiments/hp_sweep.yaml)
#   sbatch --array=0-$((N-1))%4 slurm/sweep.sh experiments/hp_sweep.yaml
# "%4" = at most 4 runs at the same time (be fair to other cluster users).
# Extra arguments after the config are passed on, e.g. --set train.seed=1
#SBATCH --job-name=eo-sweep
#SBATCH --partition=itc-gpu,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh

CONFIG=$1
shift
python -m core.cli train --experiment-config "$CONFIG" --run-index "${SLURM_ARRAY_TASK_ID}" "$@"

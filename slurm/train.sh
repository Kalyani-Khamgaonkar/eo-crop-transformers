#!/bin/bash
# One training run on the UT EEMCS HPC cluster.
#   sbatch slurm/train.sh --model vit
#   sbatch slurm/train.sh --experiment-config experiments/hp_sweep.yaml --run-index 2
#   sbatch slurm/train.sh --model vit --set train.lr=3e-4
# All arguments are passed to `python -m core.cli train`.
#SBATCH --job-name=eo-train
#SBATCH --partition=itc-gpu,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh

python -m core.cli check_gpu
python -m core.cli train "$@"

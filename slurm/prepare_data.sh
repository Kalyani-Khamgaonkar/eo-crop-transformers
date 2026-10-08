#!/bin/bash
# Data preparation on a CPU node: splits, subset, normalisation statistics.
#   sbatch slurm/prepare_data.sh /path/to/PASTIS/metadata.geojson
#SBATCH --job-name=eo-prep
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh

GEOJSON=$1
python -m core.cli make_splits --geojson "$GEOJSON" --loto
python -m core.cli make_subset --fold-run I --fraction 0.05 --seed 42
python -m core.cli compute_stats --fold-run I

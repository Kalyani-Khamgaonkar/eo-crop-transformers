# Shared job setup, sourced by every Slurm script.

module load miniconda3/25.7
eval "$(conda shell.bash hook)"
conda activate thesis

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=$HTTP_PROXY http_proxy=$HTTP_PROXY https_proxy=$HTTP_PROXY

# Paths come from .env (DATASET_DIR, OUTPUT_DIR); exported here so subprocesses see them.
if [ -f .env ]; then set -a; source .env; set +a; fi

# Optional: copy the dataset to the node's local disk for faster reading.
# Turn on with:  COPY_TO_LOCAL=1 sbatch slurm/train.sh ...
if [ "${COPY_TO_LOCAL:-0}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  LOCAL_DIR=/local/${SLURM_JOB_ID}
  mkdir -p "$LOCAL_DIR"
  echo "[env] copying $DATASET_DIR to $LOCAL_DIR"
  rsync -a "$DATASET_DIR"/ "$LOCAL_DIR"/
  export DATASET_DIR=$LOCAL_DIR
  trap 'rm -rf "$LOCAL_DIR"' EXIT   # clean up as the cluster rules ask
fi

mkdir -p logs
echo "[env] host $(hostname), job ${SLURM_JOB_ID:-none}, data $DATASET_DIR, out ${OUTPUT_DIR:-outputs}"

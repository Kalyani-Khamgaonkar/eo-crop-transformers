#!/bin/bash
# Convert ALL PASTIS patches to 24x24 pickles, make fold lists I-V, compute stats per run.
#SBATCH --job-name=eo-prep-full
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh
PASTIS_DIR=${PASTIS_DIR:-$HOME/datasets/PASTIS}
python - "$PASTIS_DIR" <<'PY'
import os, sys, pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from core.config import get_paths
from core.data.prepare import BORDER, CROP, GRID, days_since_start, read_raw_metadata, semantic_label

PASTIS = Path(sys.argv[1])
OUT = get_paths().dataset_dir / "pastis24"
OUT.mkdir(parents=True, exist_ok=True)
META = read_raw_metadata(PASTIS)

def work(pid):
    if (OUT / f"{pid}_{GRID * GRID - 1}.pickle").exists():
        return pid
    img = np.load(PASTIS / "DATA_S2" / f"S2_{pid}.npy")
    lab = semantic_label(PASTIS, pid)
    doy = days_since_start(META[pid]["dates"])
    assert len(doy) == img.shape[0], f"patch {pid}: dates/images mismatch"
    k = 0
    for i in range(GRID):
        for j in range(GRID):
            y, x = BORDER + i * CROP, BORDER + j * CROP
            with open(OUT / f"{pid}_{k}.pickle", "wb") as f:
                pickle.dump({"img": img[:, :, y:y + CROP, x:x + CROP].copy(),
                             "labels": lab[y:y + CROP, x:x + CROP].copy(), "doy": doy},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            k += 1
    return pid

with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as pool:
    for n, _ in enumerate(pool.imap_unordered(work, sorted(META), chunksize=8), 1):
        if n % 200 == 0 or n == len(META):
            print(f"converted {n}/{len(META)} patches", flush=True)
PY
python -m core.cli make_splits --geojson "$PASTIS_DIR/metadata.geojson"
for r in I II III IV V; do python -m core.cli compute_stats --fold-run $r | head -1 & done
wait
echo "[prep] done"

# EO Crop Transformers

Controlled comparison of spatial-temporal patchification and positional encoding for
transformer-based crop classification on PASTIS (Sentinel-2).

MSc thesis, M-GEO GeoAI, Faculty ITC, University of Twente.
Kalyani Khamgaonkar · Supervisors: Dr. Mahdi Farnaghi, Dr. M. Khodadadzadeh

## Research questions

- **SO1** Which patchification / representation (2D patch, tubelet, cuboid, temporal-first) gives the best mIoU and OA (RQ1.1), and at what cost in tokens, memory and time (RQ1.2)?
- **SO2** Which positional encoding (sinusoidal, learnable, relative) improves on the ViT baseline (RQ2.1) and generalises best to unseen regions (RQ2.2)?

## Design principles

1. **One protocol for every run.** `experiments/protocol.yaml` fixes data, model size, training and metrics. Experiment files may only change keys listed under `tunable`; anything else stops with an error.
2. **Models are built from parts.** A model is a combination of `tokenizer × attention × spatial_pe × temporal_enc`, not a separate file. `list_search_space` enumerates valid combinations and rejects invalid ones (divisibility, token budget, parameter budget).
3. **Same size, same head, same metrics.** Parameter count must be within ±10% of 1.7M, every model uses the same pixel head, and all metrics come from one confusion matrix.
4. **Every run is traceable.** Each run folder stores the resolved config, git commit, seed and package versions, and every finished run adds a row to `outputs/results/runs.csv`.
5. **Jobs can restart.** `last.pt` is saved every epoch and a resubmitted job resumes automatically.

## Repository layout

```
core/
  cli.py                 one entry point: python -m core.cli <command>
  config.py              protocol + overrides + paths from .env
  data/
    pastis.py            PASTIS24 dataset (time padding + mask), synthetic data, stats
    folds.py             official runs I to V, leave-one-tile-out, stratified subset
  models/
    tokenizers.py        SO1: patch2d, tubelet, cuboid
    pos_encoding.py      SO2: learnable, sinusoidal, relative; day-of-year encodings
    blocks.py            attention (with padding mask + relative bias), MLP, encoder
    builder.py           assembles a model from the design choices
    search_space.py      combination generator + constraint checks
    registry.py          named presets (vit, vit_temporal_first, vit_tubelet, vit_cuboid)
    tsvit.py             TSViT reference (wrapper around DeepSatModels)
  training/
    train.py             loop, AdamW/SGD, warmup + cosine, AMP, checkpoint + resume
    losses.py            masked cross-entropy, class weights
  evaluation/
    metrics.py           mIoU, OA, per-class IoU / recall / precision / F1
    evaluate.py          test evaluation (once per final model)
    cost.py              memory, time, inference speed
  postproc/              attention / per-tile analysis (later)
  reporting/report.py    mean ± std over seeds -> summary CSV + markdown
  utils/                 seeds, run manifest, results table
experiments/
  protocol.yaml          fixed shared protocol
  hp_sweep.yaml          hyperparameter sweep for the ViT baseline
  so1_patchification.yaml
slurm/                   train.sh, sweep.sh, prepare_data.sh, env.sh
tests/                   shapes, padding, budget, metrics, overfit check
notebooks/               quick looks only, no core logic
```

Outputs (not in git, under `OUTPUT_DIR`):

```
checkpoints/<run_name>/   config.yaml, manifest.json, best.pt, last.pt, history.json, val_metrics.json
tensorboard/<run_name>/
eval/<run_name>/          test_metrics.json, per_class_metrics.csv, confusion_matrix.png
results/runs.csv          one row per finished run (the results table)
reports/                  summary_<experiment>.csv, report_<experiment>.md
```

Run names look like `hp_sweep_vit__lr_3e-4__s42__a1b2c3`: experiment, run id, seed and a
short hash of the settings. A job only resumes a run with exactly the same settings.

## Setup

### Cluster (UT EEMCS HPC)

```bash
module load miniconda3/25.7
eval "$(conda shell.bash hook)"
conda create -n thesis python=3.11 -c conda-forge --override-channels -y
conda activate thesis
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

git clone https://github.com/<user>/<repo>.git && cd <repo>
cp .env.example .env        # then edit DATASET_DIR and OUTPUT_DIR
pytest -q                   # all tests should pass (CPU is fine)
```

Rules: edit and submit on the head node, train only through `sbatch`, never ssh into compute nodes.

### Laptop

Same steps without `module load`. The cu128 wheel works on recent NVIDIA GPUs.

## Data

1. Put the PASTIS24 pickles (from the DeepSatModels conversion) in `$DATASET_DIR/pastis24/`.
2. Check the keys of one file and, if needed, adjust `KEY_IMG`, `KEY_LABEL`, `KEY_DOY` in `core/data/pastis.py`:
   ```bash
   python -m core.cli inspect_sample $DATASET_DIR/pastis24/<some file>.pickle
   ```
3. Splits, subset and normalisation statistics (CPU job):
   ```bash
   sbatch slurm/prepare_data.sh /path/to/PASTIS/metadata.geojson
   ```
   This writes `splits/I..V/`, `splits/LOTO_<tile>/`, `subsets/0.05_s42/I/`, `stats/norm_I.json`.
   If you already have lists from your own `make_folds.py` / `make_subset.py`, copy them into the same folders instead.

## Usage

```bash
python -m core.cli --help
python -m core.cli list_models
python -m core.cli list_search_space --show-invalid

# check a config without training
python -m core.cli train --model vit --dry-run

# smoke test without PASTIS (fake data)
python -m core.cli train --model vit --synthetic --set train.epochs=2

# one real run
sbatch slurm/train.sh --model vit
sbatch slurm/train.sh --model vit --set train.lr=3e-4 --set train.weight_decay=0.05

# a whole experiment file as a job array (max 4 at once)
N=$(python -m core.cli num_runs experiments/hp_sweep.yaml)
sbatch --array=0-$((N-1))%4 slurm/sweep.sh experiments/hp_sweep.yaml

# extra seeds for the final settings
sbatch --array=0-$((N-1))%4 slurm/sweep.sh experiments/hp_sweep.yaml --set train.seed=1

# results
python -m core.cli report --experiment hp_sweep_vit

# test fold, once per final model
python -m core.cli evaluate_test <run_name>
```

Monitor: `squeue -u $USER`, `tail -f logs/<job>.out`, `seff <jobid>`,
dashboard http://hpc-status.ewi.utwente.nl/slurm/

## Workflow

1. **Sanity**: `pytest -q`, then a synthetic run, then a 2-epoch real run to time one epoch.
2. **Reproduce TSViT** on the 5% subset (~36.8% val mIoU) to confirm data and metrics match.
3. **Hyperparameter sweep** (`hp_sweep.yaml`), one change at a time; copy the best value into `base`.
4. **Freeze** the hyperparameters in the experiment files.
5. **SO1 and SO2** experiments, 3 seeds for the final configurations.
6. **Test** once per final model; leave-one-tile-out splits for RQ2.2.

## Notes

- Cuboid tokens with large blocks exceed the parameter budget (embedding and head grow with block size). `list_search_space --show-invalid` lists which ones and why.
- Joint attention with 2 × 2 patches gives 8,640 tokens per sample, above the token budget in `search_space.py` (`JOINT_TOKEN_BUDGET`). Raise it deliberately if you want to test it on the 96 GB GPUs.
- The temporal-first model uses masked mean pooling over time; TSViT uses class tokens. Keep TSViT as the external reference.
- Cluster maintenance days take the whole cluster offline; jobs resume from `last.pt` when resubmitted.

## References

Dosovitskiy et al. (2021) ViT · Tarasiou et al. (2023) TSViT · Sainte Fare Garnot & Landrieu (2021) PASTIS / U-TAE ·
Szwarcman et al. (2024) Prithvi-EO-2.0 · Gao et al. (2022) Earthformer · Shaw et al. (2018) relative position representations

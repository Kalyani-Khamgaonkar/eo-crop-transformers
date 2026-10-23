# EO Crop Transformers

Controlled comparison of spatial-temporal patchification and positional encoding for
transformer-based crop classification on PASTIS (Sentinel-2).



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


## Data

### Quick start from raw PASTIS (what we use now)

```bash
python -m core.cli prepare_subset --pastis-dir ~/datasets/PASTIS     # select 5% + convert to 24x24
python -m core.cli compute_stats --subset                            # normalisation from subset
```
Folds (Run I): train = folds 1, 2, 3; validation = fold 4; test = fold 5 (not used during tuning).
The chosen patch IDs are saved in `subsets/patch_ids_<fraction>_s<seed>_I.json`.

### Full dataset

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
2. **Overfit check** on 2 to 3 real patches: loss must go near zero.
3. **Hyperparameter sweep** for the baseline ViT on the small subset (`hp_sweep.yaml`, all runs in parallel).
4. **Combine the winners** in `baseline_final.yaml`: 100 epochs × 3 seeds. This freezes the baseline.
5. **SO1 and SO2** experiments against the baseline ViT, 3 seeds for final configurations.




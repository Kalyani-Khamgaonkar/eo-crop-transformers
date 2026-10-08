"""Command line entry point:  python -m core.cli <command> --help"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click

from core.config import get_paths, load_experiment, load_protocol, parse_cli_overrides, resolve_run


@click.group()
def cli():
    """EO crop transformers: controlled comparison of patchification and positional encoding."""


# ---------------------------------------------------------------- environment

@cli.command("check_gpu")
def check_gpu():
    """Show PyTorch, CUDA and GPU."""
    import torch

    click.echo(f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    click.echo(f"cuda available: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        click.echo(f"  gpu {i}: {p.name}, {p.total_memory / 1024 ** 3:.0f} GB")


# ---------------------------------------------------------------- models

@cli.command("list_models")
def list_models_cmd():
    """List model presets."""
    from core.models.registry import PRESETS, EXTERNAL

    for name, ov in PRESETS.items():
        short = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in ov.items())
        click.echo(f"{name:20s} {short}")
    for name, note in EXTERNAL.items():
        click.echo(f"{name:20s} {note}")


@cli.command("list_search_space")
@click.option("--csv", "csv_path", type=click.Path(), help="Also write valid configs to CSV")
@click.option("--show-invalid", is_flag=True, help="Print rejected combinations and why")
def list_search_space(csv_path, show_invalid):
    """Enumerate all valid design combinations (proposal section 4.2.3)."""
    from core.models.search_space import enumerate_configs, summary_row

    valid, invalid = enumerate_configs(load_protocol())
    rows = [summary_row(c) for c in valid]
    for r in rows:
        click.echo("  ".join(f"{k}={v}" for k, v in r.items()))
    click.echo(f"\n{len(valid) + len(invalid)} combinations, {len(valid)} valid, {len(invalid)} rejected")
    if show_invalid:
        for c, reasons in invalid:
            click.echo(f"  x {summary_row(c)} -> {'; '.join(reasons)}")
    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        click.echo(f"written {csv_path}")


# ---------------------------------------------------------------- data

@cli.command("inspect_sample")
@click.argument("path", type=click.Path(exists=True))
def inspect_sample(path):
    """Print keys, shapes and dtypes of one PASTIS24 pickle."""
    import numpy as np
    from core.data.pastis import load_pickle

    s = load_pickle(path)
    for k, v in s.items():
        a = np.asarray(v)
        click.echo(f"{k:10s} shape={a.shape} dtype={a.dtype} min={a.min():.3g} max={a.max():.3g}")


@cli.command("make_splits")
@click.option("--geojson", required=True, type=click.Path(exists=True), help="PASTIS metadata.geojson")
@click.option("--loto", is_flag=True, help="Also make leave-one-tile-out splits (RQ2.2)")
def make_splits(geojson, loto):
    """Write file lists for official runs I to V (and optionally LOTO)."""
    from core.data.folds import make_leave_one_tile_out, make_official_splits

    root = get_paths().dataset_dir
    click.echo(json.dumps(make_official_splits(root, Path(geojson)), indent=2))
    if loto:
        click.echo(json.dumps(make_leave_one_tile_out(root, Path(geojson)), indent=2))


@cli.command("make_subset")
@click.option("--fold-run", default="I")
@click.option("--fraction", default=0.05, type=float)
@click.option("--seed", default=42, type=int)
def make_subset(fold_run, fraction, seed):
    """Stratified reduced-data lists (by patch, all classes covered if possible)."""
    from core.data.folds import make_stratified_subset

    d = load_protocol()["data"]
    out = make_stratified_subset(get_paths().dataset_dir, fold_run, fraction, seed,
                                 d["num_classes"], d["ignore_index"])
    click.echo(json.dumps(out, indent=2))


@cli.command("compute_stats")
@click.option("--fold-run", default="I")
def compute_stats(fold_run):
    """Per-band mean / std from the FULL training split of a run."""
    from core.data.pastis import compute_norm_stats, split_file

    root = get_paths().dataset_dir
    stats = compute_norm_stats(root, split_file(root, fold_run, "train"), root / "stats" / f"norm_{fold_run}.json")
    click.echo(json.dumps({k: v for k, v in stats.items() if k != "n_pixels"}, indent=2))


# ---------------------------------------------------------------- training

@cli.command("num_runs")
@click.argument("experiment_config", type=click.Path(exists=True))
def num_runs(experiment_config):
    """Number of runs in an experiment file (for sbatch --array=0-(N-1))."""
    click.echo(len(load_experiment(experiment_config)["runs"]))


@cli.command("train")
@click.option("--model", "preset", help="Preset from list_models (applied before the experiment file)")
@click.option("--experiment-config", type=click.Path(exists=True), help="YAML with base + runs")
@click.option("--run-index", default=0, type=int, help="Which entry in `runs` (Slurm array index)")
@click.option("--set", "sets", multiple=True, help="Extra override, e.g. --set train.lr=3e-4")
@click.option("--synthetic", is_flag=True, help="Fake data: smoke test without PASTIS")
@click.option("--no-strict-budget", is_flag=True, help="Do not stop when params are outside the budget")
@click.option("--dry-run", is_flag=True, help="Print the resolved config and parameter count, then stop")
def train(preset, experiment_config, run_index, sets, synthetic, no_strict_budget, dry_run):
    """Train one configuration."""
    import yaml

    from core.models.builder import build_model, check_param_budget, tokens_per_sample
    from core.models.registry import preset_overrides
    from core.models.search_space import check
    from core.training.train import run_training

    overrides = preset_overrides(preset) if preset else {}
    overrides.update(parse_cli_overrides(sets))
    cfg, name = resolve_run(experiment_config, run_index, overrides)
    if preset:
        cfg["run"]["name"] = name = name.replace("adhoc", preset, 1)
        cfg["run"]["experiment"] = cfg["run"]["experiment"].replace("adhoc", preset)
    problems = check(cfg)
    if problems:
        raise click.UsageError("Invalid configuration: " + "; ".join(problems))

    if dry_run:
        click.echo(yaml.safe_dump({k: cfg[k] for k in ("run", "data", "model", "train")}, sort_keys=False))
        _, ok, msg = check_param_budget(build_model(cfg), cfg)
        click.echo(f"{msg} {'OK' if ok else 'OUTSIDE BUDGET'}, {tokens_per_sample(cfg)} tokens/sample")
        return
    run_training(cfg, synthetic=synthetic, strict_budget=not no_strict_budget)


@cli.command("evaluate_test")
@click.argument("run_name")
def evaluate_test_cmd(run_name):
    """Evaluate best.pt of a run on the test fold. Do this once, at the end."""
    from core.evaluation.evaluate import evaluate_test

    evaluate_test(run_name)


@cli.command("report")
@click.option("--experiment", help="Only this experiment name")
def report(experiment):
    """Summary CSV + markdown table from runs.csv (mean ± std over seeds)."""
    from core.reporting.report import build_report

    build_report(experiment)


if __name__ == "__main__":
    cli()

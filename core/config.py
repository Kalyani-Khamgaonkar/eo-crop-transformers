"""Configuration: fixed protocol + experiment overrides + paths from .env.

Rules
- Every run starts from experiments/protocol.yaml.
- An experiment config can only change keys listed under `tunable` in the protocol.
  Anything else raises an error, so the shared protocol cannot drift by accident.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiments" / "protocol.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    dataset_dir: Path
    output_dir: Path

    @property
    def checkpoints(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def tensorboard(self) -> Path:
        return self.output_dir / "tensorboard"

    @property
    def eval(self) -> Path:
        return self.output_dir / "eval"

    @property
    def results_csv(self) -> Path:
        return self.output_dir / "results" / "runs.csv"

    @property
    def reports(self) -> Path:
        return self.output_dir / "reports"


def get_paths() -> Paths:
    return Paths(
        dataset_dir=Path(os.environ.get("DATASET_DIR", PROJECT_ROOT / "datasets")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs")),
    )


# ---------------------------------------------------------------- dict helpers

def get_dotted(cfg: dict, key: str) -> Any:
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"Unknown config section '{part}' in '{key}'")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key '{key}'")
    node[parts[-1]] = value


# ---------------------------------------------------------------- loading

def load_protocol(path: Path | str = DEFAULT_PROTOCOL) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(protocol: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of the protocol with allowed overrides applied."""
    cfg = copy.deepcopy(protocol)
    allowed = set(protocol.get("tunable", []))
    for key, value in overrides.items():
        if key == "id":
            continue
        if key not in allowed:
            raise ValueError(
                f"'{key}' is fixed by the protocol (not in `tunable`). "
                "Agree a protocol change with the group instead of overriding it."
            )
        set_dotted(cfg, key, value)
    return cfg


def config_hash(cfg: dict, n: int = 6) -> str:
    """Short fingerprint of data/model/train settings. Same settings -> same run folder,
    so a job only resumes a run with identical settings."""
    import hashlib
    import json

    core = {k: cfg[k] for k in ("data", "model", "train") if k in cfg}
    core = json.loads(json.dumps(core))
    core["data"].pop("num_workers", None)  # does not change results
    for k in ("flex_patch", "flex_t"):     # new keys: off = absent, so old run names do not change
        if not core.get("model", {}).get(k):
            core.get("model", {}).pop(k, None)
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:n]


def load_experiment(path: Path | str) -> dict:
    with open(path) as f:
        exp = yaml.safe_load(f)
    exp.setdefault("base", {})
    exp.setdefault("runs", [{"id": "default"}])
    return exp


def resolve_run(
    experiment_path: Path | str | None = None,
    run_index: int = 0,
    extra_overrides: dict[str, Any] | None = None,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
) -> tuple[dict, str]:
    """Build the final config for one run. Returns (config, run_name)."""
    protocol = load_protocol(protocol_path)
    overrides: dict[str, Any] = {}
    exp_name, run_id = "adhoc", "default"
    if experiment_path is not None:
        exp = load_experiment(experiment_path)
        exp_name = exp.get("name", Path(experiment_path).stem)
        runs = exp["runs"]
        if not 0 <= run_index < len(runs):
            raise IndexError(f"run_index {run_index} out of range (0..{len(runs) - 1})")
        run = runs[run_index]
        run_id = str(run.get("id", f"run{run_index}"))
        overrides.update(exp["base"])
        overrides.update(run)
    if extra_overrides:
        overrides.update(extra_overrides)
    cfg = apply_overrides(protocol, overrides)
    seed = cfg["train"]["seed"]
    run_name = f"{exp_name}__{run_id}__s{seed}__{config_hash(cfg)}"
    cfg["run"] = {"experiment": exp_name, "run_id": run_id, "name": run_name}
    return cfg, run_name


def parse_cli_overrides(items: tuple[str, ...]) -> dict[str, Any]:
    """Turn ('train.lr=3e-4', 'model.attention=joint') into a dict with YAML-typed values."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, value = item.split("=", 1)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):  # PyYAML reads '3e-4' as a string
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        out[key.strip()] = parsed
    return out

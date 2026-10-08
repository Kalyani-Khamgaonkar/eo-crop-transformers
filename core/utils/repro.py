"""Reproducibility: seeds and a manifest saved with every run."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from core.config import PROJECT_ROOT


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": run("status", "--porcelain") not in ("", "unknown"),
    }


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_manifest(run_dir: Path, cfg: dict, extra: dict | None = None) -> None:
    """Write config.yaml + manifest.json into the run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    manifest = {
        "run": cfg.get("run", {}),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "env": environment_info(),
        "slurm_job_id": _slurm_job_id(),
    }
    if extra:
        manifest.update(extra)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _slurm_job_id() -> str | None:
    import os

    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{job}_{task}" if job and task else job

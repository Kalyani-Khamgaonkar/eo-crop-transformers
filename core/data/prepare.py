"""Raw PASTIS (128 x 128 patches) -> PASTIS24 pickles (24 x 24 crops) for the pipeline.

Raw PASTIS layout (Sainte Fare Garnot & Landrieu, 2021):
    metadata.geojson              ID_PATCH, Fold, TILE, dates-S2 (index -> YYYYMMDD)
    DATA_S2/S2_<id>.npy           (T, 10, 128, 128) Sentinel-2 time series
    ANNOTATIONS/TARGET_<id>.npy   (3, 128, 128); channel 0 = semantic class (19 = void)

Conversion: the central 120 x 120 px of each patch is cut into 5 x 5 = 25 crops of 24 x 24
(the 4 px border on each side is dropped). Every crop becomes one pickle:
    pastis24/<id>_<k>.pickle   {"img": (T,10,24,24) float32, "labels": (24,24) int64, "doy": (T,) int64}

`doy` is days since 2018-09-01, the start of the PASTIS season. A plain day-of-year would
repeat (Sept 2018 and Sept 2019 would get the same value), because PASTIS spans ~15 months.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from core.data.folds import OFFICIAL_RUNS

SEASON_START = date(2018, 9, 1)
CROP, GRID, BORDER = 24, 5, 4


def read_raw_metadata(pastis_dir: Path) -> dict[int, dict]:
    data = json.loads((Path(pastis_dir) / "metadata.geojson").read_text())
    out = {}
    for feat in data["features"]:
        p = feat["properties"]
        dates = p.get("dates-S2", {})
        if isinstance(dates, str):
            dates = json.loads(dates)
        ordered = [str(dates[k]) for k in sorted(dates, key=lambda x: int(x))]
        out[int(p["ID_PATCH"])] = {"fold": int(p["Fold"]), "tile": str(p.get("TILE", "")), "dates": ordered}
    return out


def days_since_start(yyyymmdd: list[str]) -> np.ndarray:
    return np.array([(date(int(d[:4]), int(d[4:6]), int(d[6:8])) - SEASON_START).days for d in yyyymmdd],
                    dtype=np.int64)


def semantic_label(pastis_dir: Path, pid: int) -> np.ndarray:
    target = np.load(Path(pastis_dir) / "ANNOTATIONS" / f"TARGET_{pid}.npy")
    return (target[0] if target.ndim == 3 else target).astype(np.int64)


def select_subset_patches(pastis_dir: Path, run: str, fraction: float, seed: int,
                          num_classes: int, ignore_index: int) -> dict[str, list[int]]:
    """Pick whole patches per split, making sure every class appears if possible."""
    meta = read_raw_metadata(pastis_dir)
    rng = np.random.default_rng(seed)
    chosen = {}
    for split in ("train", "val"):
        pids = sorted(pid for pid, m in meta.items() if m["fold"] in OFFICIAL_RUNS[run][split])
        classes = {}
        for pid in pids:
            u = np.unique(semantic_label(pastis_dir, pid))
            classes[pid] = {int(c) for c in u if c != ignore_index and c < num_classes}
        target = max(1, round(len(pids) * fraction))
        order = rng.permutation(pids).tolist()
        sel, covered = [], set()
        for c in range(num_classes):
            if c in covered:
                continue
            for pid in order:
                if pid not in sel and c in classes[pid]:
                    sel.append(pid)
                    covered |= classes[pid]
                    break
        for pid in order:
            if len(sel) >= target:
                break
            if pid not in sel:
                sel.append(pid)
        chosen[split] = sorted(sel)
        chosen[f"{split}_missing_classes"] = sorted(set(range(num_classes)) - covered)
    return chosen


def convert_patch(pastis_dir: Path, pid: int, dates: list[str], out_dir: Path) -> list[str]:
    img = np.load(Path(pastis_dir) / "DATA_S2" / f"S2_{pid}.npy").astype(np.float32)   # T,C,128,128
    lab = semantic_label(pastis_dir, pid)
    doy = days_since_start(dates)
    if len(doy) != img.shape[0]:
        raise ValueError(f"patch {pid}: {img.shape[0]} images but {len(doy)} dates")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    k = 0
    for i in range(GRID):
        for j in range(GRID):
            y, x = BORDER + i * CROP, BORDER + j * CROP
            sample = {"img": img[:, :, y:y + CROP, x:x + CROP].copy(),
                      "labels": lab[y:y + CROP, x:x + CROP].copy(),
                      "doy": doy}
            name = f"{pid}_{k}.pickle"
            with open(out_dir / name, "wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            names.append(name)
            k += 1
    return names


def prepare_subset(pastis_dir: Path, dataset_dir: Path, run: str, fraction: float, seed: int,
                   num_classes: int, ignore_index: int, include_test: bool = False) -> dict:
    """Select patches, convert them, and write the subset file lists the trainer reads."""
    from core.data.pastis import split_file

    meta = read_raw_metadata(pastis_dir)
    chosen = select_subset_patches(pastis_dir, run, fraction, seed, num_classes, ignore_index)
    if include_test:
        chosen["test"] = sorted(pid for pid, m in meta.items() if m["fold"] in OFFICIAL_RUNS[run]["test"])
    out_dir = dataset_dir / "pastis24"
    summary: dict = defaultdict(dict)
    for split in [s for s in ("train", "val", "test") if s in chosen]:
        files = []
        for pid in chosen[split]:
            files += [f"pastis24/{n}" for n in convert_patch(pastis_dir, pid, meta[pid]["dates"], out_dir)]
        lst = split_file(dataset_dir, run, split, None if split == "test" else fraction, seed)
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(files) + "\n")
        summary[split] = {"patches": len(chosen[split]), "samples": len(files), "list": str(lst),
                          "missing_classes": chosen.get(f"{split}_missing_classes", [])}
    (dataset_dir / "subsets").mkdir(exist_ok=True)
    (dataset_dir / "subsets" / f"patch_ids_{fraction:g}_s{seed}_{run}.json").write_text(
        json.dumps({k: v for k, v in chosen.items() if not k.endswith("missing_classes")}, indent=2))
    return dict(summary)

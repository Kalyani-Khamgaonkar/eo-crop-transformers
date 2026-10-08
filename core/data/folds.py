"""Train / val / test splits.

1. Official PASTIS 5-fold protocol (Sainte Fare Garnot & Landrieu, 2021): runs I to V.
2. Leave-one-tile-out (RQ2.2): the official folds all contain all 4 tiles, so they test
   new patches, not new regions. Holding out one tile tests a truly unseen region.

Both need the original PASTIS metadata.geojson (fields ID_PATCH, Fold, TILE).
If you already have file lists from your own make_folds.py, you can just copy them into
DATASET_DIR/splits/<run>/ and skip this module.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.data.pastis import _label_2d, load_pickle, read_list, KEY_LABEL

OFFICIAL_RUNS = {
    "I":   {"train": [1, 2, 3], "val": [4], "test": [5]},
    "II":  {"train": [2, 3, 4], "val": [5], "test": [1]},
    "III": {"train": [3, 4, 5], "val": [1], "test": [2]},
    "IV":  {"train": [4, 5, 1], "val": [2], "test": [3]},
    "V":   {"train": [5, 1, 2], "val": [3], "test": [4]},
}

# PASTIS24 pickles are named after their parent patch, e.g. "10000_3.pickle".
# Change this if your files are named differently.
PATCH_ID_PATTERN = re.compile(r"(\d+)")


def patch_id_of(path: str) -> int:
    m = PATCH_ID_PATTERN.search(Path(path).stem)
    if not m:
        raise ValueError(f"Cannot find a patch id in '{path}'")
    return int(m.group(1))


def read_metadata(geojson: Path) -> dict[int, dict]:
    data = json.loads(Path(geojson).read_text())
    out = {}
    for feat in data["features"]:
        p = feat["properties"]
        out[int(p["ID_PATCH"])] = {"fold": int(p["Fold"]), "tile": str(p.get("TILE", ""))}
    return out


def list_pickles(dataset_dir: Path, subdir: str = "pastis24") -> list[str]:
    root = dataset_dir / subdir
    return sorted(str(p.relative_to(dataset_dir)) for p in root.rglob("*.pickle"))


def _write(files: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(files) + "\n")


def make_official_splits(dataset_dir: Path, geojson: Path) -> dict[str, dict[str, int]]:
    meta = read_metadata(geojson)
    files = list_pickles(dataset_dir)
    summary = {}
    for run, roles in OFFICIAL_RUNS.items():
        counts = {}
        for split, folds in roles.items():
            sel = [f for f in files if meta[patch_id_of(f)]["fold"] in folds]
            _write(sel, dataset_dir / "splits" / run / f"{split}.txt")
            counts[split] = len(sel)
        summary[run] = counts
    return summary


def make_leave_one_tile_out(dataset_dir: Path, geojson: Path, val_fraction: float = 0.2,
                            seed: int = 42) -> dict[str, dict[str, int]]:
    """For each tile: test = that tile, train/val = remaining tiles split by patch."""
    meta = read_metadata(geojson)
    files = list_pickles(dataset_dir)
    tiles = sorted({m["tile"] for m in meta.values()})
    rng = np.random.default_rng(seed)
    summary = {}
    for tile in tiles:
        test = [f for f in files if meta[patch_id_of(f)]["tile"] == tile]
        rest = [f for f in files if meta[patch_id_of(f)]["tile"] != tile]
        patches = sorted({patch_id_of(f) for f in rest})
        val_patches = set(rng.choice(patches, int(len(patches) * val_fraction), replace=False).tolist())
        val = [f for f in rest if patch_id_of(f) in val_patches]
        train = [f for f in rest if patch_id_of(f) not in val_patches]
        run = f"LOTO_{tile}"
        for split, sel in (("train", train), ("val", val), ("test", test)):
            _write(sel, dataset_dir / "splits" / run / f"{split}.txt")
        summary[run] = {"train": len(train), "val": len(val), "test": len(test)}
    return summary


def make_stratified_subset(dataset_dir: Path, run: str, fraction: float, seed: int,
                           num_classes: int, ignore_index: int) -> dict[str, dict]:
    """Reduced-data lists, sampled by whole patch (all crops of a patch stay together).

    Greedy: first pick patches until every class appears, then fill randomly to `fraction`.
    """
    rng = np.random.default_rng(seed)
    out_dir = dataset_dir / "subsets" / f"{fraction:g}_s{seed}" / run
    summary = {}
    for split in ("train", "val"):
        files = read_list(dataset_dir / "splits" / run / f"{split}.txt")
        by_patch: dict[int, list[str]] = defaultdict(list)
        for f in files:
            by_patch[patch_id_of(f)].append(f)
        classes_in: dict[int, set] = {}
        for pid, fs in by_patch.items():
            present = set()
            for f in fs:
                lab = _label_2d(load_pickle(dataset_dir / f)[KEY_LABEL])
                present |= set(np.unique(lab).tolist())
            classes_in[pid] = {c for c in present if c != ignore_index and c < num_classes}

        target = max(1, round(len(by_patch) * fraction))
        order = rng.permutation(sorted(by_patch)).tolist()
        chosen, covered = [], set()
        for c in range(num_classes):  # make sure every class is covered if possible
            if c in covered:
                continue
            for pid in order:
                if pid not in chosen and c in classes_in[pid]:
                    chosen.append(pid)
                    covered |= classes_in[pid]
                    break
        for pid in order:
            if len(chosen) >= target:
                break
            if pid not in chosen:
                chosen.append(pid)
        sel = sorted(f for pid in chosen for f in by_patch[pid])
        _write(sel, out_dir / f"{split}.txt")
        summary[split] = {
            "patches": len(chosen),
            "files": len(sel),
            "missing_classes": sorted(set(range(num_classes)) - covered),
        }
    return summary

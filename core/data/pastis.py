"""PASTIS24 dataset.

Expected layout under DATASET_DIR (see .env.example):

    pastis24/            one pickle per 24 x 24 crop (from the DeepSatModels conversion)
    splits/<run>/        train.txt, val.txt, test.txt   (one pickle path per line, relative to DATASET_DIR)
    subsets/<frac>_s<seed>/<run>/train.txt, val.txt      (stratified reduced-data lists)
    stats/norm_<run>.json                                 (per-band mean / std from training data)

Each pickle is a dict. Key names are set below; check them against one of your files:
    python -m core.cli inspect_sample <path-to-pickle>

Every sample returned by the dataset is a dict:
    img   float32 (T_max, C, H, W)   normalised, zero padded in time
    doy   int64   (T_max,)           day of year, 0 for padding
    mask  bool    (T_max,)           True = real date, False = padding
    label int64   (H, W)             void / unknown = ignore_index
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Key names inside each PASTIS24 pickle (DeepSatModels convention). Adjust if yours differ.
KEY_IMG = "img"
KEY_LABEL = "labels"
KEY_DOY = "doy"


def load_pickle(path: Path | str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _label_2d(label: np.ndarray) -> np.ndarray:
    """Labels may be stored as (H, W), (1, H, W) or (H, W, k). Keep the semantic class map."""
    label = np.asarray(label)
    if label.ndim == 3:
        label = label[0] if label.shape[0] <= label.shape[-1] else label[..., 0]
    return label.astype(np.int64)


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def split_file(dataset_dir: Path, run: str, split: str,
               subset_fraction: float | None = None, subset_seed: int = 42) -> Path:
    """Where the file list for a split lives. Test is never subsampled."""
    if subset_fraction and subset_fraction < 1.0 and split != "test":
        return dataset_dir / "subsets" / f"{subset_fraction:g}_s{subset_seed}" / run / f"{split}.txt"
    return dataset_dir / "splits" / run / f"{split}.txt"


class PASTIS24(Dataset):
    def __init__(
        self,
        dataset_dir: Path | str,
        file_list: Path | str,
        max_dates: int = 60,
        ignore_index: int = 19,
        norm_stats: Path | str | None = None,
        train: bool = False,
        augment: bool = False,
    ):
        self.root = Path(dataset_dir)
        self.files = read_list(Path(file_list))
        self.max_dates = max_dates
        self.ignore_index = ignore_index
        self.train = train
        self.augment = augment and train
        self.mean = self.std = None
        if norm_stats is not None:
            stats = json.loads(Path(norm_stats).read_text())
            self.mean = np.asarray(stats["mean"], dtype=np.float32)[None, :, None, None]
            self.std = np.asarray(stats["std"], dtype=np.float32)[None, :, None, None]

    def __len__(self) -> int:
        return len(self.files)

    def _select_dates(self, t: int) -> np.ndarray:
        """Keep at most max_dates dates: random during training, evenly spaced otherwise."""
        if t <= self.max_dates:
            return np.arange(t)
        if self.train:
            return np.sort(np.random.choice(t, self.max_dates, replace=False))
        return np.linspace(0, t - 1, self.max_dates).round().astype(int)

    def __getitem__(self, idx: int) -> dict:
        s = load_pickle(self.root / self.files[idx])
        img = np.asarray(s[KEY_IMG], dtype=np.float32)          # (T, C, H, W)
        doy = np.asarray(s[KEY_DOY], dtype=np.int64).reshape(-1)
        label = _label_2d(s[KEY_LABEL])

        keep = self._select_dates(img.shape[0])
        img, doy = img[keep], doy[keep]
        if self.mean is not None:
            img = (img - self.mean) / self.std
        if self.augment:
            img, label = random_flip_rot(img, label)
        return pad_sample(img, doy, label, self.max_dates)


class SyntheticSITS(Dataset):
    """Fake data with a learnable signal, for smoke tests and the overfit check."""

    def __init__(self, n: int = 16, max_dates: int = 60, bands: int = 10, img_size: int = 24,
                 num_classes: int = 19, ignore_index: int = 19, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n):
            t = int(rng.integers(max_dates // 2, max_dates + 1))
            doy = np.sort(rng.choice(np.arange(1, 366), t, replace=False))
            # blocky label map
            coarse = rng.integers(0, num_classes, (img_size // 6, img_size // 6))
            label = np.kron(coarse, np.ones((6, 6), dtype=np.int64))
            label[rng.random(label.shape) < 0.02] = ignore_index
            phase = (label % num_classes)[None, None] / num_classes * 2 * np.pi
            season = np.sin(doy[:, None, None, None] / 365 * 2 * np.pi + phase)
            img = season + 0.1 * rng.standard_normal((t, bands, img_size, img_size))
            self.samples.append(pad_sample(img.astype(np.float32), doy, label, max_dates))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def pad_sample(img: np.ndarray, doy: np.ndarray, label: np.ndarray, max_dates: int) -> dict:
    t, c, h, w = img.shape
    out_img = np.zeros((max_dates, c, h, w), dtype=np.float32)
    out_doy = np.zeros(max_dates, dtype=np.int64)
    mask = np.zeros(max_dates, dtype=bool)
    out_img[:t], out_doy[:t], mask[:t] = img, doy, True
    return {
        "img": torch.from_numpy(out_img),
        "doy": torch.from_numpy(out_doy),
        "mask": torch.from_numpy(mask),
        "label": torch.from_numpy(np.ascontiguousarray(label)),
    }


def random_flip_rot(img: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.random.randint(4)
    img, label = np.rot90(img, k, axes=(2, 3)), np.rot90(label, k, axes=(0, 1))
    if np.random.rand() < 0.5:
        img, label = img[..., ::-1], label[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


# ---------------------------------------------------------------- statistics

def compute_norm_stats(dataset_dir: Path, file_list: Path, out_path: Path) -> dict:
    """Per-band mean / std over all real dates and pixels of the training files."""
    total, total_sq, count = None, None, 0
    for rel in read_list(file_list):
        img = np.asarray(load_pickle(dataset_dir / rel)[KEY_IMG], dtype=np.float64)  # T,C,H,W
        x = img.transpose(1, 0, 2, 3).reshape(img.shape[1], -1)
        total = x.sum(1) if total is None else total + x.sum(1)
        total_sq = (x ** 2).sum(1) if total_sq is None else total_sq + (x ** 2).sum(1)
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_pixels": int(count)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    return stats


def class_counts(dataset_dir: Path, file_list: Path, num_classes: int, ignore_index: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for rel in read_list(file_list):
        lab = _label_2d(load_pickle(dataset_dir / rel)[KEY_LABEL]).ravel()
        lab = lab[(lab != ignore_index) & (lab < num_classes)]
        counts += np.bincount(lab, minlength=num_classes)[:num_classes]
    return counts

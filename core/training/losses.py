"""Losses. Masked cross-entropy: void pixels (ignore_index) are not penalised."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def class_weights_from_counts(counts: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Inverse frequency ** power, normalised to mean 1. power=0.5 is a softer weighting.

    Classes with zero training pixels get weight 0 (they cannot be learned anyway).
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = np.zeros_like(counts)
    present = counts > 0
    w[present] = (counts[present].sum() / counts[present]) ** power
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def build_loss(cfg: dict, class_counts: np.ndarray | None = None) -> nn.Module:
    ignore = cfg["data"]["ignore_index"]
    kind = cfg["train"]["loss"]
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    if kind == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore, label_smoothing=ls)
    if kind == "weighted_ce":
        if class_counts is None:
            raise ValueError("weighted_ce needs class counts from the training set")
        return nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts), ignore_index=ignore, label_smoothing=ls)
    raise ValueError(f"Unknown loss '{kind}'")

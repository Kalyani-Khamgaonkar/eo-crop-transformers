"""All metrics come from one confusion matrix accumulated over the whole split.

    macro mIoU   mean IoU over classes (primary metric)
    OA           overall pixel accuracy (= micro accuracy)
    per class    IoU, recall (accuracy), precision, F1

A class with no true and no predicted pixels in the split is left out of the macro
average (it cannot be scored), and reported as NaN. Same rule for every model.
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int):
        self.k = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """logits (B, K, H, W), target (B, H, W)"""
        pred = logits.argmax(1).flatten()
        tgt = target.flatten()
        keep = (tgt != self.ignore_index) & (tgt >= 0) & (tgt < self.k)
        idx = tgt[keep] * self.k + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.k ** 2).view(self.k, self.k).cpu()

    def compute(self) -> dict:
        return metrics_from_confusion(self.mat.numpy())


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt = cm.sum(1)       # true pixels per class (rows = truth)
    pred = cm.sum(0)     # predicted pixels per class
    union = gt + pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        recall = np.where(gt > 0, tp / gt, np.nan)
        precision = np.where(pred > 0, tp / pred, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), np.nan)
    total = cm.sum()
    return {
        "miou_macro": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan"),
        "oa": float(tp.sum() / total) if total else float("nan"),
        "acc_macro": float(np.nanmean(recall)),
        "f1_macro": float(np.nanmean(f1)),
        "iou_micro": float(tp.sum() / union.sum()) if union.sum() else float("nan"),  # = TP / (TP+FP+FN), pooled
        "classes_scored": int(np.sum(~np.isnan(iou))),
        "per_class": {
            "iou": iou.tolist(), "recall": recall.tolist(),
            "precision": precision.tolist(), "f1": f1.tolist(),
            "support": gt.astype(int).tolist(),
        },
    }


PASTIS_CLASSES = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed",
    "Spring barley", "Sunflower", "Grapevine", "Beet", "Winter triticale", "Winter durum wheat",
    "Fruits, vegetables, flowers", "Potatoes", "Leguminous fodder", "Soybeans", "Orchard",
    "Mixed cereal", "Sorghum",
]

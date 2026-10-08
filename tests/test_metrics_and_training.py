"""Metric correctness and the 'can it overfit a tiny set' sanity check."""

import numpy as np
import torch

from core.config import apply_overrides, load_protocol
from core.data.pastis import SyntheticSITS
from core.evaluation.metrics import ConfusionMatrix, metrics_from_confusion
from core.models.builder import build_model


def test_metrics_known_values():
    # 2 classes: truth [0,0,1,1], pred [0,1,1,1]
    cm = np.array([[1, 1], [0, 2]])
    m = metrics_from_confusion(cm)
    assert np.isclose(m["oa"], 0.75)
    assert np.isclose(m["per_class"]["iou"][0], 1 / 2)
    assert np.isclose(m["per_class"]["iou"][1], 2 / 3)
    assert np.isclose(m["miou_macro"], (1 / 2 + 2 / 3) / 2)


def test_ignore_index_and_absent_class():
    cmat = ConfusionMatrix(num_classes=3, ignore_index=3)
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 0] = 1.0                                  # predicts class 0 everywhere
    target = torch.tensor([[[0, 0], [3, 1]]])           # one void pixel, class 2 absent
    cmat.update(logits, target)
    m = cmat.compute()
    assert cmat.mat.sum() == 3                          # void ignored
    assert np.isnan(m["per_class"]["iou"][2])           # absent class not scored
    assert m["classes_scored"] == 2


def test_overfit_tiny_synthetic_set():
    """Loss must fall a lot on 4 samples; if not, the pipeline has a bug."""
    torch.manual_seed(0)
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "patch2d", "model.patch_size": 4,
                                            "model.attention": "temporal_first"})
    cfg["data"]["max_dates"] = 20
    ds = SyntheticSITS(n=4, max_dates=20)
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ds[0]}
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_index"])
    first = None
    for _ in range(150):
        loss = loss_fn(model(batch["img"], batch["doy"], batch["mask"]), batch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first, f"loss {first:.3f} -> {loss.item():.3f}"

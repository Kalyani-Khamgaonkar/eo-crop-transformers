"""Pick APT-2D thresholds: tokens per sample for pairs (tau8, tau4), base patch 2 (sizes 2, 4, 8 px).

    python scripts/apt_stats.py                       # 20% fold I train, 300 batches
    python scripts/apt_stats.py --synthetic           # code check only

Prints entropy percentiles per region size, then for each (tau8, tau4) pair: tokens per sample and
the share of the image covered by 8, 4 and 2 px tokens. Reference budgets with 60 dates:
patch 4 = 2,160 tokens, patch 3 = 3,840, patch 6 = 960 (fewer real dates -> fewer tokens).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from core.config import resolve_run
from core.models.apt import quadtree, region_entropy
from core.training.train import build_loaders

QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--max-batches", type=int, default=300)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cfg, _ = resolve_run(None, extra_overrides={
        "data.subset_fraction": a.fraction, "data.num_workers": 4, "train.loss": "ce", "train.augment": False})
    train_dl, _, _ = build_loaders(cfg, synthetic=a.synthetic)

    batches, e8, e4 = [], [], []
    for i, batch in enumerate(train_dl):
        if i >= a.max_batches:
            break
        img, mask = batch["img"], batch["mask"]
        batches.append((img, mask))
        e8.append(region_entropy(img, mask, 8).flatten())
        e4.append(region_entropy(img, mask, 4).flatten())
    e8, e4 = torch.cat(e8), torch.cat(e4)
    q8 = [torch.quantile(e8, q).item() for q in QS]
    q4 = [torch.quantile(e4, q).item() for q in QS]
    print(f"\n{len(batches) * cfg['train']['batch_size']} samples, {a.fraction:g} of fold I train")
    print("entropy (bits) of 8 px regions: " + "  ".join(f"p{int(q*100)} {v:.2f}" for q, v in zip(QS, q8)))
    print("entropy (bits) of 4 px regions: " + "  ".join(f"p{int(q*100)} {v:.2f}" for q, v in zip(QS, q4)))
    print(f"\n{'tau8':>6} {'tau4':>6} {'tokens/sample':>14} {'8px area':>9} {'4px area':>9} {'2px area':>9}")
    for t8 in q8:
        for t4 in q4:
            tok, area, n = 0.0, torch.zeros(3), 0
            for img, mask in batches:
                sel = quadtree(img, mask, 2, [t4, t8])                  # finest first: sizes 2, 4, 8 px
                base_cells = sel[0][0].numel()                          # 2 px cells in the image (144)
                per_date = sum(s.flatten(1).sum(1).float() for s in sel)
                tok += (per_date * mask.sum(1)).sum().item()
                for l, s in enumerate(sel):                              # a level-l token covers 4**l cells
                    area[l] += (s.flatten(1).sum(1).float() * 4 ** l / base_cells).sum()
                n += img.shape[0]
            area = 100 * area / n
            print(f"{t8:6.2f} {t4:6.2f} {tok / n:14.0f} {area[2]:8.0f}% {area[1]:8.0f}% {area[0]:8.0f}%")
    print("\nArea = share of the image covered by tokens of that size. "
          "Pick a pair near 2,160 tokens (= patch 4 budget).")


if __name__ == "__main__":
    main()

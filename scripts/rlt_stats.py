"""How many tokens does RLT keep for each threshold tau? Use it to pick rlt_tau before training.

    python scripts/rlt_stats.py                      # 20% fold I train list, patch 4, 300 batches
    python scripts/rlt_stats.py --fraction 1.0 --max-batches 500
    python scripts/rlt_stats.py --synthetic          # code check only

Prints, per tau: share of real tokens kept, kept tokens per sample, attention cost relative to
no RLT (tokens squared), mean run length. Also prints percentiles of the date-to-date change
per patch, so you can see where the thresholds sit. No augmentation, no GPU needed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from core.config import resolve_run
from core.models.rlt import find_runs, patchify
from core.training.train import build_loaders

TAUS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=300)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cfg, _ = resolve_run(None, extra_overrides={
        "data.subset_fraction": a.fraction, "data.num_workers": 4, "train.loss": "ce", "train.augment": False})
    train_dl, _, _ = build_loaders(cfg, synthetic=a.synthetic)

    real_tok, kept = 0, {t: 0 for t in TAUS}
    sq_base, sq = 0.0, {t: 0.0 for t in TAUS}
    runs = {t: 0 for t in TAUS}
    diffs, n_samples = [], 0
    for i, batch in enumerate(train_dl):
        if i >= a.max_batches:
            break
        img, doy, mask = batch["img"], batch["doy"], batch["mask"]
        b, t = mask.shape
        n = (img.shape[-1] // a.patch) ** 2
        real = mask.sum(1).float() * n                                 # real tokens per sample
        real_tok += real.sum().item()
        sq_base += (real ** 2).sum().item()
        n_samples += b
        pt = patchify(img.float(), a.patch)
        both = (mask[:, 1:] & mask[:, :-1])[:, :, None].expand(b, t - 1, n)
        d = (pt[:, 1:] - pt[:, :-1]).abs().mean(-1)[both]
        diffs.append(d[torch.randperm(d.numel())[:5000]])
        for tau in TAUS:
            keep, _, _ = find_runs(img, doy, mask, a.patch, tau)
            k = keep.flatten(1).sum(1).float()
            kept[tau] += k.sum().item()
            sq[tau] += (k ** 2).sum().item()
            runs[tau] += k.sum().item()
    d = torch.cat(diffs)
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\n{n_samples} samples, patch {a.patch}, {a.fraction:g} of fold I train, "
          f"{real_tok / n_samples:.0f} real tokens per sample on average")
    print("date-to-date change per patch (mean |dx|, normalised bands), percentiles:")
    print("  " + "  ".join(f"p{int(q * 100)} {torch.quantile(d, q).item():.3f}" for q in qs))
    print(f"\n{'tau':>6} {'kept %':>8} {'tokens/sample':>14} {'attn cost':>10} {'mean run (dates)':>17}")
    for tau in TAUS:
        print(f"{tau:6.2f} {100 * kept[tau] / real_tok:7.1f}% {kept[tau] / n_samples:14.0f} "
              f"{sq[tau] / sq_base:9.2f}x {real_tok / max(runs[tau], 1):17.2f}")
    print("\nattn cost = sum(tokens^2) relative to no RLT. Pick a tau that keeps about 40 to 60% of tokens "
          "for a clear speed-up; check the run lengths stay short enough to keep the crop calendar.")


if __name__ == "__main__":
    main()

"""Score each flex model at every patch size / tubelet length it was trained with (validation set).

    python scripts/eval_flex.py experiments/flex_6nov.yaml

Uses best.pt of each run (picked on validation at the base size). The test fold is not touched.
Writes OUTPUT_DIR/reports/flex_eval.csv and prints a table.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from core.config import get_paths, load_experiment, resolve_run
from core.models.builder import build_model
from core.training.train import build_loaders, evaluate


class AtSize(torch.nn.Module):
    def __init__(self, model, patch=None, tlen=None):
        super().__init__()
        self.model, self.patch, self.tlen = model, patch, tlen

    def forward(self, img, doy, mask):
        return self.model(img, doy, mask, patch=self.patch, tlen=self.tlen)


def main():
    exp_path = sys.argv[1]
    synthetic = "--synthetic" in sys.argv          # code check only, fake data
    paths = get_paths()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for i in range(len(load_experiment(exp_path)["runs"])):
        cfg, name = resolve_run(exp_path, i)
        ck = paths.checkpoints / name / "best.pt"
        if not ck.exists():
            print(f"{name}: no best.pt yet, skipped"); continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model = build_model(state["config"]).to(dev)
        model.load_state_dict(state["model"])
        _, val_dl, _ = build_loaders(state["config"], synthetic=synthetic)
        m = state["config"]["model"]
        sizes = [("patch", s) for s in m.get("flex_patch") or []] + [("tlen", s) for s in m.get("flex_t") or []]
        H, T = cfg["data"]["img_size"], cfg["data"]["max_dates"]
        for kind, s in sizes:
            wrap = AtSize(model, **{kind: s})
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            r = evaluate(wrap, val_dl, None, dev, state["config"])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            p = s if kind == "patch" else m["patch_size"]
            t = s if kind == "tlen" else (m["tubelet_t"] if m["tokenizer"] != "patch2d" else 1)
            rows.append(dict(run=cfg["run"]["run_id"], size=f"{t}x{p}x{p}", tokens=(T // t) * (H // p) ** 2,
                             val_miou=round(r["miou_macro"], 4), val_oa=round(r["oa"], 4),
                             val_seconds=round(time.time() - t0, 1), best_epoch=state["epoch"]))
            print(f"{rows[-1]['run']:16} size {rows[-1]['size']:7} tokens {rows[-1]['tokens']:5d}  "
                  f"val mIoU {rows[-1]['val_miou']:.4f}  OA {rows[-1]['val_oa']:.4f}  ({rows[-1]['val_seconds']} s)")
    if rows:
        out = paths.output_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "flex_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nsaved {out / 'flex_eval.csv'}")
    print("\nFixed-size references (5%, fold I, val): 2D p=4 sin spatial + date sin (case 1, seed 42) 0.456; "
          "baseline t=1 0.454 (3 seeds); tubelet t=2 + date sin 0.427 (3 seeds)")


if __name__ == "__main__":
    main()

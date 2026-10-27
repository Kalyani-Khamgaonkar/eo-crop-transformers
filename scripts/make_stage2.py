"""Stage 2 grid: top-2 Stage 1 configs x regularisation (weighted CE fixed).
Usage: python scripts/make_stage2.py "sgd,1e-1,5e-4" "adamw,1e-2,0.1"
"""
import sys
from sklearn.model_selection import ParameterGrid
configs = [c.split(",") for c in sys.argv[1:]]
assert len(configs) == 2, "give exactly 2 configs: optimiser,lr,wd"
reg = ParameterGrid({"model.dropout": [0.0, 0.1, 0.2], "model.drop_path": [0.0, 0.1, 0.2],
                     "train.augment": [False, "flip", "flip_date"], "train.label_smoothing": [0.0, 0.1]})
base = {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1, "model.attention": "joint",
        "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin", "train.loss": "weighted_ce",
        "data.subset_fraction": 0.05, "data.num_workers": 6}
y = lambda v: "false" if v is False else v
runs = []
for g in reg:                      # config innermost: both configs always compared at the same settings
    for o, lr, wd in configs:
        runs.append({"train.optimizer": o, "train.lr": float(lr), "train.weight_decay": float(wd), **g})
keys = {tuple(sorted(r.items())) for r in runs}
assert len(runs) == len(keys) == 2 * len(reg) == 108
with open("experiments/hp_stage2.yaml", "w") as f:
    f.write("# Stage 2: top-2 Stage 1 configs x dropout x drop path x augmentation x label smoothing (ParameterGrid)\n")
    f.write("name: hp_stage2\n\nbase:\n")
    for k, v in base.items(): f.write(f"  {k}: {v}\n")
    f.write("\nruns:\n")
    for n, r in enumerate(runs):
        f.write(f"  - {{id: s2_{n:03d}_{r['train.optimizer']}, " +
                ", ".join(f"{k}: {y(v) if not isinstance(v, float) else format(v, '.1e')}" for k, v in r.items()) + "}\n")
print("written experiments/hp_stage2.yaml:", len(runs), "runs")

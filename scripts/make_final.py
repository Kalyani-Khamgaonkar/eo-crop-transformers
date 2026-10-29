"""Pick the Stage 2 winner + top 3 and write experiments/hp_final.yaml (fold I full data + Stage 3 + Stage 4)."""
import re, glob, yaml
runs = yaml.safe_load(open("experiments/hp_stage2.yaml"))["runs"]
EP = re.compile(r"\[ep\s+(\d+)\] loss [\d.]+/[\d.]+\s+mIoU ([\d.]+)/([\d.]+)")
KEYS = ["train.optimizer", "train.lr", "train.weight_decay", "model.dropout", "model.drop_path", "train.augment", "train.label_smoothing"]
res = []
for f in glob.glob("logs/run_hp_stage2_[0-9]*.out"):
    i = int(re.search(r"_(\d+)\.out$", f).group(1)); t = open(f).read()
    m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
    if not m: continue
    ep = {int(a): (float(b), float(c)) for a, b, c in EP.findall(t)}; b = int(m.group(2))
    gap = ep[b][0] - ep[b][1] if b in ep else 9.0
    res.append(dict(run=i, val=float(m.group(1)), gap=gap, cfg={k: runs[i][k] for k in KEYS}))
print(f"Stage 2 runs finished: {len(res)} / 108")
best = max(d["val"] for d in res)
tied = [d for d in res if d["val"] >= best - 0.01]
winner = min(tied, key=lambda d: d["gap"])
others = sorted([d for d in res if d is not winner], key=lambda d: -d["val"])[:2]
top3 = [winner] + others
show = lambda d: f"run {d['run']:3d}  val {d['val']:.3f}  gap {d['gap']:.3f}  " + "  ".join(f"{k.split('.')[1]}={d['cfg'][k]}" for k in KEYS)
print(f"\nBest val {best:.3f}; {len(tied)} runs within 0.01 (tie) -> smallest gap wins")
print("WINNER  " + show(winner))
for d in others: print("TOP 3   " + show(d))
base = {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1, "model.attention": "joint",
        "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin", "train.loss": "weighted_ce",
        "data.subset_fraction": 0.05, "data.num_workers": 6}
fmt = lambda v: "false" if v is False else (format(v, ".1e") if isinstance(v, float) else v)
line = lambda rid, d: "  - {id: " + rid + ", " + ", ".join(f"{k}: {fmt(v)}" for k, v in d.items()) + "}\n"
out = []
out.append(line("fold1_full", {**winner["cfg"], "data.subset_fraction": 1.0, "data.fold_run": "I"}))
for sch in ["cosine", "step", "constant"]:
    for wu in [0, 5, 10]:
        if (sch, wu) == ("cosine", 10): continue          # = the winner itself (already run in Stage 2)
        out.append(line(f"s3_{sch}_wu{wu}", {**winner["cfg"], "train.scheduler": sch, "train.warmup_epochs": wu}))
for k, d in enumerate(top3, 1):
    for seed in [1, 2]:
        out.append(line(f"s4_top{k}_seed{seed}", {**d["cfg"], "train.seed": seed}))
assert len(out) == 15
with open("experiments/hp_final.yaml", "w") as f:
    f.write("# Final night: fold I full data (Stage 2 winner) + Stage 3 schedule + Stage 4 seeds\nname: hp_final\n\nbase:\n")
    for k, v in base.items(): f.write(f"  {k}: {v}\n")
    f.write("\nruns:\n" + "".join(out))
print("\nwritten experiments/hp_final.yaml: 15 runs")

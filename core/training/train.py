"""Training loop shared by every configuration.

- AdamW or SGD, linear warmup then cosine decay to min_lr (per step)
- masked (optionally class-weighted) cross-entropy
- bf16 mixed precision on GPU
- saves last.pt every epoch and best.pt on best validation mIoU
- resumes automatically from last.pt (Slurm time limits, maintenance days)
- writes config.yaml, manifest.json, history.json, val_metrics.json and one row in runs.csv
- never touches the test split (use `evaluate_test` once at the end)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import get_paths
from core.data.pastis import PASTIS24, SyntheticSITS, class_counts, split_file
from core.evaluation.cost import Timer, peak_memory_gb, reset_peak_memory
from core.evaluation.metrics import ConfusionMatrix
from core.models.builder import build_model, check_param_budget, tokens_per_sample
from core.training.losses import build_loss
from core.utils.repro import git_info, save_manifest, set_seed
from core.utils.results import append_result, row_from_config


# ---------------------------------------------------------------- data

def build_loaders(cfg: dict, synthetic: bool = False, overfit: int = 0):
    d, t = cfg["data"], cfg["train"]
    if synthetic:
        kw = dict(max_dates=d["max_dates"], bands=d["bands"], img_size=d["img_size"],
                  num_classes=d["num_classes"], ignore_index=d["ignore_index"])
        train_ds, val_ds = SyntheticSITS(n=16, seed=0, **kw), SyntheticSITS(n=8, seed=1, **kw)
        counts = None
    else:
        root = get_paths().dataset_dir
        run = d["fold_run"]
        stats = root / "stats" / f"norm_{run}.json"
        if not stats.exists():
            raise FileNotFoundError(
                f"{stats} missing. Run: python -m core.cli compute_stats --fold-run {run}"
            )
        lists = {s: split_file(root, run, s, d["subset_fraction"], d["subset_seed"]) for s in ("train", "val")}
        for s, p in lists.items():
            if not p.exists():
                raise FileNotFoundError(f"{s} list {p} missing. See README: make_splits / make_subset")
        common = dict(max_dates=d["max_dates"], ignore_index=d["ignore_index"], norm_stats=stats)
        train_ds = PASTIS24(root, lists["train"], train=True, augment=t["augment"] and not overfit, **common)
        val_ds = PASTIS24(root, lists["val"], train=False, **common)
        if overfit:  # sanity check: same few files for training and validation
            train_ds.files = train_ds.files[:overfit]
            val_ds = PASTIS24(root, lists["train"], train=False, **common)
            val_ds.files = train_ds.files
        counts = None
        if t["loss"] == "weighted_ce":
            counts = class_counts(root, lists["train"], d["num_classes"], d["ignore_index"])

    nw = 0 if synthetic else d["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=nw,
                          pin_memory=torch.cuda.is_available(), drop_last=len(train_ds) > t["batch_size"])
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    return train_dl, val_dl, counts


# ---------------------------------------------------------------- optimisation

def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    # no weight decay on biases, norms, positional embeddings
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "pe" in name.split(".")[-1] or "table" in name else decay).append(p)
    groups = [{"params": decay, "weight_decay": t["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]))
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=t["lr"], momentum=t["momentum"], nesterov=True)
    raise ValueError(f"Unknown optimizer '{t['optimizer']}'")


def build_scheduler(opt, cfg: dict, steps_per_epoch: int):
    t = cfg["train"]
    total = max(1, t["epochs"] * steps_per_epoch)
    warm = t["warmup_epochs"] * steps_per_epoch
    floor = t["min_lr"] / t["lr"]

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


# ---------------------------------------------------------------- loops

def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, loss_fn, opt, sched, device, cfg) -> dict:
    model.train()
    cm = ConfusionMatrix(cfg["data"]["num_classes"], cfg["data"]["ignore_index"])
    total_loss, n = 0.0, 0
    for batch in loader:
        img, doy, mask, label = (batch[k].to(device, non_blocking=True) for k in ("img", "doy", "mask", "label"))
        with _autocast(device, cfg["train"]["amp"]):
            logits = model(img, doy, mask)
            loss = loss_fn(logits.float(), label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["train"]["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        sched.step()
        total_loss += loss.item() * img.shape[0]
        n += img.shape[0]
        cm.update(logits.detach(), label)
    m = cm.compute()
    return {"loss": total_loss / max(n, 1), "miou": m["miou_macro"], "oa": m["oa"]}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg) -> dict:
    model.eval()
    cm = ConfusionMatrix(cfg["data"]["num_classes"], cfg["data"]["ignore_index"])
    total_loss, n = 0.0, 0
    for batch in loader:
        img, doy, mask, label = (batch[k].to(device, non_blocking=True) for k in ("img", "doy", "mask", "label"))
        with _autocast(device, cfg["train"]["amp"]):
            logits = model(img, doy, mask)
        if loss_fn is not None:
            total_loss += loss_fn(logits.float(), label).item() * img.shape[0]
        n += img.shape[0]
        cm.update(logits, label)
    m = cm.compute()
    m["loss"] = total_loss / max(n, 1)
    m["confusion_matrix"] = cm.mat.tolist()
    return m


# ---------------------------------------------------------------- main entry

def run_training(cfg: dict, synthetic: bool = False, strict_budget: bool = True, overfit: int = 0) -> dict:
    paths = get_paths()
    run_dir = paths.checkpoints / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    n_params, ok, msg = check_param_budget(model, cfg)
    print(f"[model] {msg}, {tokens_per_sample(cfg)} tokens/sample")
    if not ok and strict_budget:
        raise ValueError(f"Parameter budget violated: {msg}. Use --no-strict-budget only for debugging.")

    train_dl, val_dl, counts = build_loaders(cfg, synthetic, overfit)
    loss_fn = build_loss(cfg, counts).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(train_dl))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(paths.tensorboard / cfg["run"]["name"])
    except Exception:
        pass

    start_epoch, best, history = 0, {"miou": -1.0, "epoch": -1}, []
    last_ckpt = run_dir / "last.pt"
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, best, history = state["epoch"] + 1, state["best"], state["history"]
        print(f"[resume] from epoch {start_epoch}")
    else:
        save_manifest(run_dir, cfg, {"params": n_params, "tokens_per_sample": tokens_per_sample(cfg),
                                     "synthetic": synthetic})

    reset_peak_memory()
    run_timer = Timer()
    epoch_times = []
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = Timer()
        tr = train_one_epoch(model, train_dl, loss_fn, opt, sched, device, cfg)
        va = evaluate(model, val_dl, loss_fn, device, cfg)
        epoch_times.append(t0.elapsed())
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_miou": tr["miou"], "train_oa": tr["oa"],
               "val_loss": va["loss"], "val_miou": va["miou_macro"], "val_oa": va["oa"],
               "sec": epoch_times[-1]}
        history.append(row)
        print(f"[ep {epoch:3d}] loss {tr['loss']:.3f}/{va['loss']:.3f}  "
              f"mIoU {tr['miou']:.3f}/{va['miou_macro']:.3f}  OA {tr['oa']:.3f}/{va['oa']:.3f}  "
              f"{epoch_times[-1]:.0f}s")
        if writer:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)

        if va["miou_macro"] > best["miou"]:
            best = {"miou": va["miou_macro"], "epoch": epoch, "oa": va["oa"], "train_oa": tr["oa"]}
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, run_dir / "best.pt")
            (run_dir / "val_metrics.json").write_text(json.dumps({"epoch": epoch, **va}, indent=2))
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "best": best,
                    "history": history}, last_ckpt)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    if writer:
        writer.close()

    summary = {
        **row_from_config(cfg),
        "best_epoch": best["epoch"],
        "val_miou": round(best["miou"], 4),
        "val_oa": round(best.get("oa", float("nan")), 4),
        "train_oa": round(best.get("train_oa", float("nan")), 4),
        "train_val_gap": round(best.get("train_oa", float("nan")) - best.get("oa", float("nan")), 4),
        "params": n_params,
        "tokens_per_sample": tokens_per_sample(cfg),
        "peak_mem_gb": round(peak_memory_gb(), 2),
        "sec_per_epoch": round(float(np.mean(epoch_times)), 1) if epoch_times else "",
        "total_hours": round(run_timer.elapsed() / 3600, 3),
        "git_commit": git_info()["commit"][:10],
        "slurm_job_id": json.loads((run_dir / "manifest.json").read_text()).get("slurm_job_id"),
    }
    if epoch_times and not synthetic and not overfit:  # only log real runs
        append_result(paths.results_csv, summary)
    print(f"[done] best val mIoU {best['miou']:.4f} at epoch {best['epoch']}  -> {run_dir}")
    return summary

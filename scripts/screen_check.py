"""Build check for the 6 Nov screening (pe_screen_6nov.yaml) plus the reference cells.
Real model, params (soft), tokens, forward + backward with 10 padded dates, padding-invariance test.

    python scripts/screen_check.py              # batch 2, all configs, T = 60 (GPU)
    python scripts/screen_check.py 2 all 12     # T = 12: quick logic check on CPU

Hard criteria (a FAIL): grid divisibility and PE shape (search_space.check), build error,
wrong output shape, missing / non-finite gradients, padding changes the output.
Soft (reported only): parameter budget, token count.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml

from core.config import apply_overrides, load_protocol
from core.models.builder import build_model, check_param_budget, tokens_per_sample
from core.models.search_space import check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configs():
    """One entry per config id (seed 42 run) from the screening file, plus the reference cells."""
    out = {}
    for f in ("pe_screen_6nov.yaml", "nine_rest.yaml"):
        exp = yaml.safe_load(open(os.path.join(ROOT, "experiments", f)))
        for r in exp["runs"]:
            cid = r["id"].rsplit("_s", 1)[0]
            if f == "nine_rest.yaml" and cid not in ("C2_2d_conv", "C5_tube2_conv"):
                continue
            out.setdefault(cid, {**exp["base"], **{k: v for k, v in r.items() if k != "id"}})
    out["baseline_2d_E1"] = {**out["E1_tube"], "model.tokenizer": "patch2d", "model.tubelet_t": 1}
    return out


def batch(cfg, B, dev, pad=10):
    d = cfg["data"]; T, C, H = d["max_dates"], d["bands"], d["img_size"]
    pad = min(pad, T // 3) // 2 * 2                    # even, so tubelets never mix real and padded dates
    x = torch.randn(B, T, C, H, H, device=dev)
    doy = torch.zeros(B, T, dtype=torch.long, device=dev)
    doy[:, :T - pad] = torch.sort(torch.randint(0, 456, (B, T - pad), device=dev), dim=1).values
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev); mask[:, :T - pad] = True
    x[~mask] = 0.0
    return x, doy, mask


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cfgs = configs()
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(cfgs)
    T = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}, batch {B}, max_dates {T or 60}\n")
    print(f"{'id':16} {'params':>10} {'budget':7} {'tokens':>6} {'out':16} {'grads':5} {'pad-inv':7} {'peak GB':>7}  result")
    n_fail = 0
    for cid in only:
        try:
            cfg = apply_overrides(load_protocol(), cfgs[cid])
            if T:
                cfg["data"]["max_dates"] = T
            hard = check(cfg)
            if hard:
                raise ValueError("hard criteria: " + "; ".join(hard))
            torch.manual_seed(0)
            model = build_model(cfg).to(dev)
            n, ok, _ = check_param_budget(model, cfg)
            x, doy, mask = batch(cfg, B, dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.train()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                out = model(x, doy, mask)
            out.float().mean().backward()
            bad = sum(1 for p in model.parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()))
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
            model.eval()
            with torch.no_grad():
                y1 = model(x, doy, mask)
                x2 = x.clone(); x2[~mask] = torch.randn_like(x2[~mask]) * 5
                y2 = model(x2, doy, mask)
            pad_ok = torch.allclose(y1, y2, atol=1e-4)
            K, H = cfg["data"]["num_classes"], cfg["data"]["img_size"]
            good = tuple(out.shape) == (B, K, H, H) and not bad and pad_ok
            n_fail += not good
            print(f"{cid:16} {n:10,d} {'ok' if ok else 'soft!':7} {tokens_per_sample(cfg):6d} {str(tuple(out.shape)):16} "
                  f"{'ok' if not bad else bad:5} {'ok' if pad_ok else 'LEAKS':7} {peak:7.2f}  {'OK' if good else 'FAIL'}")
        except Exception as e:
            n_fail += 1
            print(f"{cid:16} FAIL {type(e).__name__}: {e}")
    print(f"\n{len(only) - n_fail} / {len(only)} OK   (budget 'soft!' = outside 1.7M +/- 10%, reported only)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

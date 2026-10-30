"""Build check for the 9 configurations (and the baseline): real model, params, tokens,
forward + backward on a dummy batch with 10 padded dates, padding-invariance test.
Usage: python scripts/nine_check.py [batch] [ids, e.g. C7,C8,C9] [max_dates]
A smaller max_dates (e.g. 12) checks the logic with far less memory (CPU); the full 60 needs a GPU for C7 to C9."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.config import load_protocol, apply_overrides
from core.models.builder import build_model, check_param_budget, tokens_per_sample

FIX = {"model.attention": "joint", "model.patch_size": 4, "model.dropout": 0.1, "model.drop_path": 0.1}
ADD_LEARN = {"model.pe_mode": "add", "model.spatial_pe": "learnable", "model.temporal_enc": "doy_learned"}
CONV = {"model.pe_mode": "conv", "model.spatial_pe": "none", "model.temporal_enc": "doy_sin"}
CONCAT = {"model.pe_mode": "concat", "model.spatial_pe": "none", "model.temporal_enc": "doy_sin"}
TOK = {"2d": {"model.tokenizer": "patch2d", "model.tubelet_t": 1},
       "tube": {"model.tokenizer": "tubelet", "model.tubelet_t": 2},
       "groups": {"model.tokenizer": "groups", "model.tubelet_t": 1}}
NINE = {
    "C1": {**TOK["2d"], **ADD_LEARN},   "C2": {**TOK["2d"], **CONV},   "C3": {**TOK["2d"], **CONCAT},
    "C4": {**TOK["tube"], **ADD_LEARN}, "C5": {**TOK["tube"], **CONV}, "C6": {**TOK["tube"], **CONCAT},
    "C7": {**TOK["groups"], **ADD_LEARN, "model.group_enc": "learned"},
    "C8": {**TOK["groups"], **CONV, "model.group_enc": "learned"},
    "C9": {**TOK["groups"], **CONCAT, "model.group_enc": "sin"},
}
EXPECTED = {"C1": 2160, "C2": 2160, "C3": 2160, "C4": 1080, "C5": 1080, "C6": 1080, "C7": 6480, "C8": 6480, "C9": 6480}


def batch(cfg, B, dev, pad=10):
    d = cfg["data"]; T, C, H = d["max_dates"], d["bands"], d["img_size"]
    pad = min(pad, T // 3)
    x = torch.randn(B, T, C, H, H, device=dev)
    doy = torch.zeros(B, T, dtype=torch.long, device=dev)
    doy[:, :T - pad] = torch.sort(torch.randint(0, 456, (B, T - pad), device=dev), dim=1).values
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev); mask[:, :T - pad] = True
    x[~mask] = 0.0
    return x, doy, mask


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(NINE)
    T = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}, batch {B}\n")
    print(f"{'id':4} {'params':>10} {'budget':6} {'tokens':>6} {'exp':>5} {'out':16} {'grads':5} {'pad-inv':8} {'peak GB':>7}  result")
    for cid, ov in ((k, v) for k, v in NINE.items() if k in only):
        try:
            cfg = apply_overrides(load_protocol(), {**FIX, **ov})
            if T: cfg["data"]["max_dates"] = T
            torch.manual_seed(0)
            model = build_model(cfg).to(dev)
            n, ok, msg = check_param_budget(model, cfg)
            tok = tokens_per_sample(cfg)
            x, doy, mask = batch(cfg, B, dev)
            if dev == "cuda": torch.cuda.reset_peak_memory_stats()
            model.train()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                out = model(x, doy, mask)
            out.float().mean().backward()
            bad = sum(1 for p in model.parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()))
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
            # padding invariance: changing values on padded dates must not change the output (eval mode, fp32)
            model.eval()
            with torch.no_grad():
                y1 = model(x, doy, mask)
                x2 = x.clone(); x2[~mask] = torch.randn_like(x2[~mask]) * 5
                y2 = model(x2, doy, mask)
            pad_ok = torch.allclose(y1, y2, atol=1e-4)
            K, H = cfg["data"]["num_classes"], cfg["data"]["img_size"]
            exp = EXPECTED[cid] * cfg["data"]["max_dates"] // 60
            good = ok and tuple(out.shape) == (B, K, H, H) and not bad and tok == exp
            print(f"{cid:4} {n:10,d} {'ok' if ok else 'OUT':6} {tok:6d} {exp:5d} {str(tuple(out.shape)):16} "
                  f"{'ok' if not bad else bad:5} {'ok' if pad_ok else 'LEAKS':8} {peak:7.2f}  {'OK' if good else 'CHECK'}")
            if not ok: print("     budget:", msg)
        except Exception as e:
            import traceback
            print(f"{cid:4} ERROR {type(e).__name__}: {e}"); traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()

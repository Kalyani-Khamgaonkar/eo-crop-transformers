"""Build check: instantiate each runnable configuration with the repo code, count params,
run one forward + backward pass on a dummy padded batch. Usage: python scripts/build_check.py [batch]"""
import sys, inspect, time, traceback, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.config import load_protocol, apply_overrides
from core.models.builder import build_model, check_param_budget, tokens_per_sample

FINAL = {"model.attention": "joint", "model.dropout": 0.1, "model.drop_path": 0.1}
CASES = {
    "baseline": {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1, "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin"},
    "case1":    {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1, "model.spatial_pe": "sinusoidal", "model.temporal_enc": "doy_sin"},
    "case2":    {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.tubelet_t": 1, "model.spatial_pe": "learnable", "model.temporal_enc": "index_learned"},
    "case3":    {"model.tokenizer": "tubelet", "model.patch_size": 4, "model.tubelet_t": 2, "model.spatial_pe": "learnable", "model.temporal_enc": "index_learned"},
    "case4":    {"model.tokenizer": "cuboid",  "model.patch_size": 4, "model.tubelet_t": 4, "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin"},
}
EXPECTED_TOKENS = {"baseline": 2160, "case1": 2160, "case2": 2160, "case3": 1080, "case4": 540}

def dummy_batch(cfg, B, device):
    d = cfg["data"]; T, C, H = d["max_dates"], d["bands"], d["img_size"]
    x = torch.randn(B, T, C, H, H, device=device)
    real = T - 10                                              # last 10 dates are padding
    doy = torch.zeros(B, T, dtype=torch.long, device=device)
    doy[:, :real] = torch.sort(torch.randint(0, 456, (B, real), device=device), dim=1).values
    mask = torch.zeros(B, T, dtype=torch.bool, device=device); mask[:, :real] = True
    return x, doy, mask

def call(model, x, doy, mask):
    names = list(inspect.signature(model.forward).parameters)
    pool = {"x": x, "img": x, "images": x, "inp": x, "doy": doy, "dates": doy, "mask": mask, "date_mask": mask, "pad_mask": mask}
    if names == ["batch"]:
        return model({"img": x, "x": x, "doy": doy, "mask": mask})
    return model(*[pool[n] for n in names if n in pool])

def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}, batch {B}\n")
    print(f"{'case':9} {'params':>10} {'budget':6} {'tokens':>6} {'exp':>5} {'out shape':18} {'grads':6} {'s/step':>6} {'peak GB':>7}  result")
    for name, ov in CASES.items():
        try:
            cfg = apply_overrides(load_protocol(), {**FINAL, **ov})
            torch.manual_seed(0)
            model = build_model(cfg).to(dev).train()
            n, ok, msg = check_param_budget(model, cfg)
            tok = tokens_per_sample(cfg)
            x, doy, mask = dummy_batch(cfg, B, dev)
            if dev == "cuda": torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                out = call(model, x, doy, mask)
            out.float().mean().backward()
            dt = time.time() - t0
            K, H = cfg["data"]["num_classes"], cfg["data"]["img_size"]
            shape_ok = tuple(out.shape) == (B, K, H, H)
            bad = [p for p in model.parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
            res = "OK" if (ok and shape_ok and not bad and tok == EXPECTED_TOKENS[name]) else "CHECK"
            print(f"{name:9} {n:10,d} {'ok' if ok else 'OUT':6} {tok:6d} {EXPECTED_TOKENS[name]:5d} {str(tuple(out.shape)):18} "
                  f"{'ok' if not bad else str(len(bad))+' bad':6} {dt:6.2f} {peak:7.2f}  {res}")
            if not ok: print("   budget:", msg)
            if bad: print("   params with missing / non-finite grads:", len(bad), "(unused parameters or nan)")
        except Exception as e:
            print(f"{name:9} ERROR: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)

if __name__ == "__main__":
    main()

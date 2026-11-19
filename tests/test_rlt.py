"""RLT (run-length tokens over time). Run: pytest -q tests/test_rlt.py"""

import pytest
import torch

from core.config import apply_overrides, config_hash, load_protocol
from core.models.builder import build_model
from core.models.rlt import find_runs

BASE = {"model.tokenizer": "patch2d", "model.patch_size": 4, "model.attention": "joint",
        "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin"}


def _batch(cfg, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = cfg["data"]
    t = d["max_dates"]
    img = torch.randn(b, t, d["bands"], d["img_size"], d["img_size"], generator=g)
    doy = torch.sort(torch.randint(1, 450, (b, t), generator=g), dim=1).values
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, t // 2:] = False
    doy[~mask] = 0
    img[~mask] = 0
    return img, doy, mask


def test_no_merge_equals_plain_joint_model():
    """tau tiny + no length encoding: RLT drops only padding, so the output must match the plain model."""
    plain_cfg = apply_overrides(load_protocol(), BASE)
    rlt_cfg = apply_overrides(load_protocol(), {**BASE, "model.rlt_tau": 1e-12, "model.rlt_len_enc": "none"})
    torch.manual_seed(0)
    plain = build_model(plain_cfg).eval()
    rlt = build_model(rlt_cfg).eval()
    rlt.load_state_dict(plain.state_dict(), strict=True)
    img, doy, mask = _batch(plain_cfg)
    with torch.no_grad():
        a, b = plain(img, doy, mask), rlt(img, doy, mask)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_runs_found_on_repeated_dates():
    img = torch.randn(1, 6, 2, 8, 8)
    img[:, 1:4] = img[:, 0:1]                       # dates 0-3 identical
    doy = torch.tensor([[10, 20, 30, 40, 50, 0]])
    mask = torch.tensor([[True] * 5 + [False]])
    keep, runlen, d = find_runs(img, doy, mask, p=4, tau=1e-6, date="start")
    assert keep[0, :, 0].tolist() == [True, False, False, False, True, False]
    assert runlen[0, :, 0].tolist() == [4, 0, 0, 0, 1, 0]
    assert d[0, 0, 0] == 10 and d[0, 4, 0] == 50
    _, _, dm = find_runs(img, doy, mask, p=4, tau=1e-6, date="mid")
    assert dm[0, 0, 0] == 25                         # mean of 10, 20, 30, 40


def test_merge_backward_and_fewer_tokens():
    cfg = apply_overrides(load_protocol(), {**BASE, "model.rlt_tau": 10.0})   # huge tau: one run per patch
    model = build_model(cfg).train()
    img, doy, mask = _batch(cfg)
    out = model(img, doy, mask)
    d = cfg["data"]
    assert out.shape == (2, d["num_classes"], d["img_size"], d["img_size"])
    out.mean().backward()
    assert model.rlt_len.weight.grad is not None
    assert float(model.rlt_keep) < 0.1               # one token per patch out of 30-60 dates


@pytest.mark.parametrize("te", ["doy_sin", "doy_learned"])
@pytest.mark.parametrize("date", ["start", "mid"])
def test_variants_run(te, date):
    cfg = apply_overrides(load_protocol(), {**BASE, "model.temporal_enc": te, "model.rlt_tau": 1.0,
                                            "model.rlt_date": date})
    model = build_model(cfg).eval()
    img, doy, mask = _batch(cfg)
    assert torch.isfinite(model(img, doy, mask)).all()


def test_old_run_names_unchanged():
    cfg = apply_overrides(load_protocol(), BASE)
    off = apply_overrides(load_protocol(), {**BASE, "model.rlt_tau": 0, "model.rlt_date": "mid"})
    assert config_hash(cfg) == config_hash(off)

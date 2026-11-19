"""APT-2D (adaptive patch sizes by entropy). Run: pytest -q tests/test_apt.py"""

import torch

from core.config import apply_overrides, config_hash, load_protocol
from core.models.apt import expand, quadtree, region_entropy
from core.models.builder import build_model

BASE = {"model.tokenizer": "patch2d", "model.patch_size": 2, "model.attention": "joint",
        "model.spatial_pe": "learnable", "model.temporal_enc": "doy_sin"}


def _batch(cfg, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = cfg["data"]
    t = 12
    img = torch.randn(b, t, d["bands"], d["img_size"], d["img_size"], generator=g)
    img[:, :, :, :8, :8] = 0.3                                   # one flat 8 x 8 block
    doy = torch.sort(torch.randint(1, 450, (b, t), generator=g), dim=1).values
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, t // 2:] = False
    doy[~mask] = 0
    img[~mask] = 0
    return img, doy, mask


def test_entropy_of_flat_region_is_zero():
    img = torch.zeros(1, 2, 3, 8, 8)
    img[..., 4:, 4:] = torch.randn(1, 2, 3, 4, 4)
    e = region_entropy(img, torch.ones(1, 2, dtype=torch.bool), 4)
    assert e[0, 0, 0] == 0 and e[0, 1, 1] > 0


def test_every_pixel_covered_once():
    img = torch.randn(3, 5, 10, 24, 24)
    img[:, :, :, :8] = 0.0
    mask = torch.ones(3, 5, dtype=torch.bool)
    for taus in ([0.5, 2.0], [5.0, 0.1], [9.0, 9.0], [-1.0, -1.0]):
        sel = quadtree(img, mask, 2, taus)
        cover = sum(expand(s.long()[..., None], 2 ** l)[..., 0] for l, s in enumerate(sel))
        assert (cover == 1).all(), taus


def test_no_merge_equals_plain_patch2_model():
    plain_cfg = apply_overrides(load_protocol(), BASE)
    apt_cfg = apply_overrides(load_protocol(), {**BASE, "model.apt_tau": [-1.0, -1.0]})   # never merge
    torch.manual_seed(0)
    plain = build_model(plain_cfg).eval()
    apt = build_model(apt_cfg).eval()
    missing, unexpected = apt.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("tokenizer.merge") or k.startswith("tokenizer.zero") for k in missing)
    img, doy, mask = _batch(plain_cfg)
    with torch.no_grad():
        a, b = plain(img, doy, mask), apt(img, doy, mask)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_merge_fewer_tokens_and_backward():
    cfg = apply_overrides(load_protocol(), {**BASE, "model.apt_tau": [9.0, 9.0]})          # always merge to 8 px
    model = build_model(cfg).train()
    img, doy, mask = _batch(cfg)
    out = model(img, doy, mask)
    d = cfg["data"]
    assert out.shape == (2, d["num_classes"], d["img_size"], d["img_size"])
    out.mean().backward()
    assert model.tokenizer.zero[1].weight.grad.abs().sum() > 0
    assert float(model.apt_tokens) <= 9 * 12                        # 9 tokens per date


def test_mixed_sizes_run():
    cfg = apply_overrides(load_protocol(), {**BASE, "model.apt_tau": [3.0, 3.5], "model.spatial_pe": "sinusoidal"})
    model = build_model(cfg).eval()
    img, doy, mask = _batch(cfg)
    assert torch.isfinite(model(img, doy, mask)).all()


def test_old_run_names_unchanged():
    cfg = apply_overrides(load_protocol(), BASE)
    off = apply_overrides(load_protocol(), {**BASE, "model.apt_tau": []})
    assert config_hash(cfg) == config_hash(off)

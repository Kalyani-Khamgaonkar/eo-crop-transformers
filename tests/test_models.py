"""Shape, budget and search-space checks. Run: pytest -q"""

import pytest
import torch

from core.config import apply_overrides, load_protocol
from core.models.builder import build_model, check_param_budget
from core.models.registry import PRESETS
from core.models.search_space import check, enumerate_configs


def _batch(cfg, b=2):
    d = cfg["data"]
    t = d["max_dates"]
    img = torch.randn(b, t, d["bands"], d["img_size"], d["img_size"])
    doy = torch.randint(1, 366, (b, t))
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, t // 2:] = False  # padding in the first sample
    doy[~mask] = 0
    return img, doy, mask


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_shape_and_budget(preset):
    cfg = apply_overrides(load_protocol(), PRESETS[preset])
    model = build_model(cfg).eval()
    img, doy, mask = _batch(cfg)
    out = model(img, doy, mask)
    d = cfg["data"]
    assert out.shape == (2, d["num_classes"], d["img_size"], d["img_size"])
    assert torch.isfinite(out).all()
    _, ok, msg = check_param_budget(model, cfg)
    assert ok, msg


@pytest.mark.parametrize("spe", ["learnable", "sinusoidal", "relative", "none"])
@pytest.mark.parametrize("att", ["joint", "temporal_first"])
def test_positional_encodings(spe, att):
    cfg = apply_overrides(load_protocol(), {
        "model.tokenizer": "patch2d", "model.patch_size": 4, "model.attention": att, "model.spatial_pe": spe,
    })
    out = build_model(cfg).eval()(*_batch(cfg))
    assert torch.isfinite(out).all()


def test_padding_does_not_change_output():
    """Values in padded dates must not influence predictions."""
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "patch2d", "model.patch_size": 4,
                                            "model.attention": "temporal_first"})
    model = build_model(cfg).eval()
    img, doy, mask = _batch(cfg, b=1)
    mask[0, 30:] = False
    a = model(img, doy, mask)
    img2 = img.clone()
    img2[0, 30:] = 999.0
    b = model(img2, doy, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_search_space_rejects_invalid():
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "patch2d", "model.patch_size": 5})
    assert check(cfg), "patch 5 does not divide 24"
    valid, invalid = enumerate_configs(load_protocol())
    assert valid and invalid


def test_protocol_blocks_fixed_keys():
    with pytest.raises(ValueError):
        apply_overrides(load_protocol(), {"train.batch_size": 32})

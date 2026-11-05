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
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "tubelet", "model.patch_size": 4, "model.tubelet_t": 7})
    assert check(cfg), "60 dates do not divide into tubelets of 7"
    valid, invalid = enumerate_configs(load_protocol())
    assert valid
    assert all(any(r.startswith(("grid", "PE shape")) for r in reasons) for _, reasons in invalid)


def test_budget_and_tokens_are_soft():
    """Agreed 5 Nov: only grid divisibility and PE shape reject; budget and token count only warn."""
    from core.models.search_space import soft_warnings
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "cuboid", "model.patch_size": 12,
                                            "model.tubelet_t": 2, "model.attention": "joint"})
    assert not check(cfg), "large cuboid breaks the parameter budget but must not be rejected"
    assert any(w.startswith("params") for w in soft_warnings(cfg))
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "patch2d", "model.patch_size": 2, "model.attention": "joint"})
    assert not check(cfg), "8,640 tokens is over the token budget but must not be rejected"
    assert any(w.startswith("tokens") for w in soft_warnings(cfg))


@pytest.mark.parametrize("ov", [
    {"model.pe_mode": "add", "model.spatial_pe": "none", "model.temporal_enc": "none"},
    {"model.pe_mode": "concat", "model.spatial_pe": "none", "model.temporal_enc": "index_sin"},
    {"model.pe_mode": "rel3d", "model.spatial_pe": "none", "model.temporal_enc": "none"},
])
def test_new_encodings_build_and_ignore_padding(ov):
    import torch
    from core.models.builder import build_model
    cfg = apply_overrides(load_protocol(), {"model.tokenizer": "tubelet", "model.patch_size": 4, "model.tubelet_t": 2,
                                            "model.attention": "joint", **ov})
    cfg["data"]["max_dates"] = 8
    model = build_model(cfg).eval()
    x = torch.randn(2, 8, 10, 24, 24); doy = torch.arange(8).repeat(2, 1) * 10
    mask = torch.ones(2, 8, dtype=torch.bool); mask[:, 6:] = False; x[~mask] = 0
    with torch.no_grad():
        y1 = model(x, doy, mask)
        x[~mask] = 5.0
        y2 = model(x, doy, mask)
    assert y1.shape == (2, 19, 24, 24) and torch.allclose(y1, y2, atol=1e-4)


def test_learned_date_table_starts_small():
    from core.models.pos_encoding import TemporalEncoding
    for kind in ("doy_learned", "index_learned"):
        assert TemporalEncoding(kind, 128, 60).emb.weight.std() < 0.03


def test_protocol_blocks_fixed_keys():
    with pytest.raises(ValueError):
        apply_overrides(load_protocol(), {"train.batch_size": 32})

"""TSViT reference (Tarasiou et al., CVPR 2023).

For now TSViT runs from your patched DeepSatModels clone (it already gives 36.8% mIoU on
the 5% subset). Two options to bring it into this repo later:

1. Wrapper (quick): keep DeepSatModels next to this repo and import its model.
       export DEEPSAT_DIR=/home/s3295303/DeepSatModels
   then `build_tsvit(cfg)` below loads it, so it trains with this repo's loop, metrics
   and results table. Its forward takes a single tensor with the day-of-year channel
   appended, so the adapter builds that input from (img, doy).

2. Port (clean): copy the TSViT classes into this file and cite the original repo
   (license: check DeepSatModels LICENSE before copying).

Option 1 is enough for the meeting; switch to 2 before the final experiments so every
model shares exactly the same training and evaluation code.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


class TSViTAdapter(nn.Module):
    def __init__(self, tsvit: nn.Module):
        super().__init__()
        self.tsvit = tsvit

    def forward(self, img: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # DeepSatModels expects (B, T, H, W, C+1) with the last channel = doy / 365.
        # Verify this against your converted data before trusting results.
        b, t, c, h, w = img.shape
        x = img.permute(0, 1, 3, 4, 2)
        doy_ch = (doy.float() / 365.0)[:, :, None, None, None].expand(b, t, h, w, 1)
        return self.tsvit(torch.cat([x, doy_ch], dim=-1))


def build_tsvit(model_config: dict) -> nn.Module:
    deepsat = os.environ.get("DEEPSAT_DIR")
    if not deepsat:
        raise RuntimeError("Set DEEPSAT_DIR to your DeepSatModels clone to use TSViT here")
    sys.path.insert(0, deepsat)
    from models.TSViT.TSViTdense import TSViT  # type: ignore  # noqa: E402

    return TSViTAdapter(TSViT(model_config))

"""FlexiViT-style tokenizer: one set of weights for several patch sizes and tubelet lengths.

Beyer et al., "FlexiViT: One Model for All Patch Sizes", CVPR 2023.
The patch embedding is stored once at the base size (t0 x p0 x p0). For another size (t x p x p)
it is resized with PI-resize: the new weights w' solve  <B x, w'> = <x, w>  for every input x,
where B is the bilinear (here trilinear) resize from t0 x p0 x p0 to t x p x p, so
w' = pinv(B^T) w. At the base size B is the identity, so the base model is unchanged.

    x (B, T, C, H, W) -> tokens (B, T/t, (H/p)(W/p), D), plus the mean date and mask per time token
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.tokenizers import _group_time


def pi_resize_matrix(src: tuple[int, int, int], dst: tuple[int, int, int], device) -> torch.Tensor:
    """(n_dst, n_src) matrix M with  w_dst = M @ w_src  (flattened t x p x p kernels)."""
    n0 = src[0] * src[1] * src[2]
    basis = torch.eye(n0, device=device).view(n0, 1, *src)
    bt = F.interpolate(basis, size=dst, mode="trilinear", align_corners=False).reshape(n0, -1)  # = B^T
    return torch.linalg.pinv(bt)                                                                # (n_dst, n_src)


class FlexTokenizer(nn.Module):
    def __init__(self, in_ch: int, dim: int, p0: int, t0: int):
        super().__init__()
        self.p0, self.t0 = p0, t0
        self.weight = nn.Parameter(torch.empty(dim, in_ch, t0, p0, p0))
        self.bias = nn.Parameter(torch.empty(dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))          # same start as nn.Conv3d
        bound = 1 / math.sqrt(in_ch * t0 * p0 * p0)
        nn.init.uniform_(self.bias, -bound, bound)
        self._mats: dict = {}

    def kernel(self, t: int, p: int) -> torch.Tensor:
        if (t, p) == (self.t0, self.p0):
            return self.weight
        key = (t, p, self.weight.device)
        if key not in self._mats:
            self._mats[key] = pi_resize_matrix((self.t0, self.p0, self.p0), (t, p, p), self.weight.device)
        d, c = self.weight.shape[:2]
        w = self.weight.reshape(d * c, -1).float() @ self._mats[key].T
        return w.reshape(d, c, t, p, p).to(self.weight.dtype)

    def forward(self, x, doy, mask, t: int, p: int):
        b, T, c, h, w = x.shape
        if T % t or h % p or w % p:
            raise ValueError(f"flex size t={t}, p={p} does not divide T={T}, H={h}, W={w}")
        tok = F.conv3d(x.permute(0, 2, 1, 3, 4), self.kernel(t, p), self.bias, stride=(t, p, p))  # (B, D, T', h', w')
        tok = tok.permute(0, 2, 3, 4, 1)
        tok = tok.reshape(b, tok.shape[1], -1, tok.shape[-1])                                        # (B, T', N, D)
        if t == 1:
            return tok, doy, mask
        doy_t, mask_t = _group_time(doy, mask, t)
        return tok, doy_t, mask_t

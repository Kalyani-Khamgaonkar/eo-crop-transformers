"""Transformer building blocks (pre-norm, as in ViT)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth: randomly skip the whole residual branch per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        x        (B, L, D)
        key_mask (B, L)  True = real token, False = padding (ignored as key)
        bias     (h, L, L) additive attention bias (relative positional encoding)
        """
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if bias is not None:
            attn_mask = bias.unsqueeze(0).to(q.dtype)
        if key_mask is not None:
            pad = torch.zeros(b, 1, 1, n, dtype=q.dtype, device=x.device)
            pad = pad.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            attn_mask = pad if attn_mask is None else attn_mask + pad
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, key_mask=None, bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), key_mask, bias))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        rates = torch.linspace(0, drop_path, depth).tolist() if depth > 1 else [drop_path]
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout, r) for r in rates)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_mask=None, bias=None):
        for blk in self.blocks:
            x = blk(x, key_mask, bias)
        return self.norm(x)

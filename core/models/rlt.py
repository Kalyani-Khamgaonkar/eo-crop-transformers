"""Run-Length Tokenization (RLT) over time for satellite image time series.

Choudhury et al., "Don't Look Twice: Faster Video Transformers with Run-Length Tokenization",
NeurIPS 2024. RLT finds patches that repeat over time, keeps only the first one of each run and
tells the model how long the run is.

Here, per spatial patch (p x p pixels, all bands):
    diff_t  = mean | x_t - x_(t-1) |      over bands and pixels, on normalised values
    repeat  : diff_t < tau and both dates are real  -> the token at t joins the current run
    run     : one kept token, the date of its start (or the run midpoint), and its length k
The model sees only the kept tokens (padding is removed by compaction), so attention cost
drops with the number of repeated patches. For the pixel head, each kept token stands for the
k dates of its run: the mean over time is weighted by k (APT does the same by repeating tokens).

    tau = 0  -> nothing merges, the model equals the plain joint model (tested)
"""

from __future__ import annotations

import torch


def patchify(img: torch.Tensor, p: int) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T, N, C*p*p), N in row-major order (same as Conv2d + flatten)."""
    b, t, c, h, w = img.shape
    gh, gw = h // p, w // p
    x = img.reshape(b, t, c, gh, p, gw, p).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(b, t, gh * gw, c * p * p)


@torch.no_grad()
def find_runs(img: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor, p: int, tau: float,
              date: str = "start") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    img (B, T, C, H, W) normalised, doy (B, T), mask (B, T) True = real date.
    Returns, each (B, T, N):
        keep    True for the first token of every run (only real dates)
        runlen  number of dates in the run, at kept positions (0 elsewhere)
        doy_tok date given to each kept token: run start, or run mean ("mid"), rounded
    """
    pt = patchify(img.float(), p)                                     # (B, T, N, F)
    b, t, n, _ = pt.shape
    real = mask[:, :, None].expand(b, t, n)
    rep = torch.zeros_like(real)
    if t > 1:
        diff = (pt[:, 1:] - pt[:, :-1]).abs().mean(-1)                # (B, T-1, N)
        rep[:, 1:] = real[:, 1:] & real[:, :-1] & (diff < tau)
    keep = real & ~rep
    run_id = (keep.long().cumsum(1) - 1).clamp(min=0)                 # run index along time
    realf = real.float()
    length = torch.zeros(b, t, n, device=img.device).scatter_add_(1, run_id, realf)
    runlen = length.gather(1, run_id) * keep                          # length at the run start
    d = doy[:, :, None].expand(b, t, n).float()
    if date == "mid":
        dsum = torch.zeros(b, t, n, device=img.device).scatter_add_(1, run_id, d * realf)
        doy_tok = (dsum.gather(1, run_id) / length.gather(1, run_id).clamp(min=1)).round()
    elif date == "start":
        doy_tok = d
    else:
        raise ValueError(f"Unknown rlt_date '{date}' (start | mid)")
    return keep, runlen.long(), (doy_tok * keep).long()


def compact(tok: torch.Tensor, keep: torch.Tensor, runlen: torch.Tensor):
    """Gather the kept tokens of each sample to the front.
    tok (B, T, N, D), keep / runlen (B, T, N)
    Returns x (B, L, D), key_mask (B, L), loc (B, L) spatial index, weight (B, L) run length,
    with L = the largest number of kept tokens in the batch."""
    b, t, n, d = tok.shape
    k = keep.reshape(b, t * n)
    n_keep = k.sum(1)
    L = int(n_keep.max())
    order = torch.sort((~k).to(torch.int8), dim=1, stable=True).indices[:, :L]   # kept first, time order kept
    x = tok.reshape(b, t * n, d).gather(1, order[..., None].expand(b, L, d))
    key_mask = torch.arange(L, device=tok.device)[None] < n_keep[:, None]
    loc = order % n
    weight = runlen.reshape(b, t * n).gather(1, order).to(tok.dtype) * key_mask
    return x, key_mask, loc, weight


def weighted_pool(x: torch.Tensor, loc: torch.Tensor, weight: torch.Tensor, n: int) -> torch.Tensor:
    """Mean over time per spatial location, each run token counted run-length times.
    x (B, L, D) -> (B, N, D)"""
    b, L, d = x.shape
    w = weight.to(x.dtype)
    num = torch.zeros(b, n, d, device=x.device, dtype=x.dtype).scatter_add_(1, loc[..., None].expand(b, L, d), x * w[..., None])
    den = torch.zeros(b, n, device=x.device, dtype=x.dtype).scatter_add_(1, loc, w)
    return num / den.clamp(min=1.0)[..., None]

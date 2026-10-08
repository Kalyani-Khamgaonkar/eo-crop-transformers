"""Computational cost (RQ1.2): parameters, tokens, GPU memory, time."""

from __future__ import annotations

import time

import torch


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - self.start


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


@torch.no_grad()
def inference_time_per_sample(model, batch: dict, device, repeats: int = 20) -> float:
    """Seconds per sample for a forward pass (after warm-up)."""
    model.eval()
    img, doy, mask = (batch[k].to(device) for k in ("img", "doy", "mask"))
    for _ in range(3):
        model(img, doy, mask)
    t = Timer()
    for _ in range(repeats):
        model(img, doy, mask)
    return t.elapsed() / (repeats * img.shape[0])

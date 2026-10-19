"""Latency measurement: warm-up runs, then timed runs (mean / median / p95, throughput)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _stats(times_s: list[float], batch_size: int) -> dict:
    ms = np.asarray(times_s) * 1000.0
    return {"batch_size": batch_size, "runs": len(ms), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "throughput_samples_per_s": float(batch_size / (ms.mean() / 1000.0))}


@torch.no_grad()
def model_latency(model: torch.nn.Module, window: int, feature_dim: int, batch_size: int, device: torch.device,
                  warmup: int, runs: int) -> dict:
    """Forward pass only, on a tensor already on the device."""
    model = model.to(device).eval()
    x = torch.randn(batch_size, window, feature_dim, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {**_stats(times, batch_size), "device": device.type, "warmup": warmup, "timing": "model forward only"}


def callable_latency(fn: Callable[[], object], batch_size: int, runs: int, warmup: int, timing: str) -> dict:
    """Times an arbitrary zero-argument callable (e.g. load + resample + assemble + model)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {**_stats(times, batch_size), "warmup": warmup, "timing": timing}

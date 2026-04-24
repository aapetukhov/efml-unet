from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.metrics.ssim import SSIMMetric


def measure_latency(
    model: nn.Module,
    device: str,
    input_size: tuple = (1, 3, 256, 256),
    n_warmup: int = 20,
    n_runs: int = 200,
) -> float:
    """Average forward-pass latency in milliseconds."""
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_warmup):
                model(dummy)
            torch.cuda.synchronize()
            start.record()
            for _ in range(n_runs):
                model(dummy)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / n_runs
        else:
            for _ in range(n_warmup):
                model(dummy)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(dummy)
            return (time.perf_counter() - t0) * 1000 / n_runs


def measure_ssim(
    model: nn.Module,
    val_loader,
    device: str,
    max_batches: int | None = None,
) -> float:
    metric = SSIMMetric()
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr).clamp(0.0, 1.0)
            for pred, tgt in zip(sr.cpu(), hr.cpu()):
                total += float(metric(pred, tgt))
                count += 1
    return total / max(count, 1)


def model_sparsity(model: nn.Module) -> float:
    """Fraction of zero-valued weights across all parameters."""
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p.data == 0).sum())
    return zeros / max(total, 1)


def save_results(results: dict, results_dir: str, name: str) -> Path:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path

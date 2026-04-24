#!/usr/bin/env python3
"""Global magnitude pruning of SRUNetHeavy + optional finetune + benchmark.

Usage:
  python scripts/run_prune.py
  python scripts/run_prune.py prune_ratio=0.4 results_name=prune_40
  python scripts/run_prune.py prune_ratio=0.5 finetune.epochs=10 results_name=prune_50
  python scripts/run_prune.py finetune.enabled=false results_name=prune_30_no_ft
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from src.acceleration.benchmark import measure_latency, measure_ssim, model_sparsity, save_results
from src.acceleration.prune import apply_global_magnitude_pruning, make_pruning_permanent
from src.datasets.data_utils import get_dataloaders
from src.utils import select_device, set_random_seed

log = logging.getLogger(__name__)


def _finetune(model, cfg, device, train_loader):
    criterion = instantiate(cfg.finetune.criterion).to(device)
    optimizer = instantiate(cfg.finetune.optimizer, params=model.parameters())
    model.train()
    for epoch in range(1, cfg.finetune.epochs + 1):
        total_loss, n = 0.0, 0
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(lr), hr)["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        log.info(f"  finetune {epoch}/{cfg.finetune.epochs} | loss: {total_loss / n:.4f}")


@hydra.main(version_base=None, config_path="../src/configs", config_name="channel_prune")
def main(cfg: DictConfig) -> None:
    set_random_seed(42)
    device = select_device("auto")
    log.info(f"Device: {device}")

    model = instantiate(cfg.model).to(device)
    ckpt_path = to_absolute_path(cfg.checkpoint_in)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded: {ckpt_path}")

    dataloaders, _ = get_dataloaders(cfg, device)
    val_loader = dataloaders.get("val") or dataloaders.get("test")

    # Baseline
    log.info("Measuring baseline...")
    baseline_ssim = measure_ssim(model, val_loader, device)
    baseline_latency = measure_latency(
        model, device, tuple(cfg.benchmark.input_size),
        cfg.benchmark.n_warmup, cfg.benchmark.n_runs,
    )
    log.info(f"Baseline | SSIM: {baseline_ssim:.4f} | Latency: {baseline_latency:.2f} ms")

    # Apply pruning
    log.info(f"Pruning prune_ratio={cfg.prune_ratio}...")
    apply_global_magnitude_pruning(model, cfg.prune_ratio)

    if cfg.finetune.enabled:
        log.info(f"Finetuning {cfg.finetune.epochs} epochs (masks still active)...")
        _finetune(model, cfg, device, dataloaders["train"])

    # Bake masks into weights for clean checkpoint + correct sparsity count
    make_pruning_permanent(model)
    sparsity = model_sparsity(model)
    log.info(f"Effective sparsity after pruning: {sparsity:.3f}")

    # Result
    result_ssim = measure_ssim(model, val_loader, device)
    result_latency = measure_latency(
        model, device, tuple(cfg.benchmark.input_size),
        cfg.benchmark.n_warmup, cfg.benchmark.n_runs,
    )
    log.info(f"Result  | SSIM: {result_ssim:.4f} | Latency: {result_latency:.2f} ms")
    log.info(f"Speedup: {baseline_latency / result_latency:.3f}x | SSIM drop: {baseline_ssim - result_ssim:.4f}")

    # Save checkpoint
    out_path = to_absolute_path(cfg.checkpoint_out)
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    log.info(f"Checkpoint: {out_path}")

    results = {
        "method": "global_magnitude_prune",
        "timestamp": datetime.now().isoformat(),
        "checkpoint_in": cfg.checkpoint_in,
        "checkpoint_out": cfg.checkpoint_out,
        "baseline": {"ssim": round(baseline_ssim, 4), "latency_ms": round(baseline_latency, 2)},
        "result":   {"ssim": round(result_ssim, 4),   "latency_ms": round(result_latency, 2)},
        "speedup":   round(baseline_latency / result_latency, 3),
        "ssim_drop": round(baseline_ssim - result_ssim, 4),
        "sparsity":  round(sparsity, 4),
        "prune_ratio": cfg.prune_ratio,
        "finetuned": cfg.finetune.enabled,
        "finetune_epochs": cfg.finetune.epochs if cfg.finetune.enabled else 0,
    }
    path = save_results(results, to_absolute_path(cfg.results_dir), cfg.results_name)
    log.info(f"Results: {path}")


if __name__ == "__main__":
    main()

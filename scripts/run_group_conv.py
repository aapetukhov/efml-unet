#!/usr/bin/env python3
"""Replace 1x1 expand/project convs with grouped convs + finetune + benchmark.

Groups=G means each output filter looks at only in_ch/G input channels.
FLOPs in targeted layers are divided by G.  Gives real speedup on any GPU.

Usage:
  python scripts/run_group_conv.py
  python scripts/run_group_conv.py groups=4 target=both results_name=group4_both
  python scripts/run_group_conv.py groups=2 target=expand finetune.epochs=10
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

from src.acceleration.benchmark import measure_latency, measure_ssim, save_results
from src.acceleration.group_conv import convert_to_grouped_conv
from src.datasets.data_utils import get_dataloaders
from src.utils import select_device, set_random_seed

log = logging.getLogger(__name__)


def _count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


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


@hydra.main(version_base=None, config_path="../src/configs", config_name="group_conv")
def main(cfg: DictConfig) -> None:
    set_random_seed(42)
    device = select_device("auto")
    log.info(f"Device: {device}")

    model = instantiate(cfg.model).to(device)
    ckpt_path = to_absolute_path(cfg.checkpoint_in)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded: {ckpt_path}  ({_count_params(model):.1f}M params)")

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

    # Convert to grouped
    log.info(f"Converting to grouped conv (G={cfg.groups}, target={cfg.target})...")
    block_log = convert_to_grouped_conv(model, groups=cfg.groups, target=cfg.target)
    for name, desc in block_log.items():
        log.info(f"  {name}: {desc}")
    log.info(f"After conversion: {_count_params(model):.1f}M params")

    # Verify forward pass works before finetuning
    dummy = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        _ = model(dummy)
    log.info("Forward pass OK")

    if cfg.finetune.enabled:
        log.info(f"Finetuning {cfg.finetune.epochs} epochs...")
        _finetune(model, cfg, device, dataloaders["train"])

    # Result
    result_ssim = measure_ssim(model, val_loader, device)
    result_latency = measure_latency(
        model, device, tuple(cfg.benchmark.input_size),
        cfg.benchmark.n_warmup, cfg.benchmark.n_runs,
    )
    log.info(f"Result  | SSIM: {result_ssim:.4f} | Latency: {result_latency:.2f} ms")
    log.info(f"Speedup: {baseline_latency / result_latency:.3f}x | SSIM drop: {baseline_ssim - result_ssim:.4f}")

    out_path = to_absolute_path(cfg.checkpoint_out)
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    log.info(f"Checkpoint: {out_path}")

    results = {
        "method": "group_conv",
        "timestamp": datetime.now().isoformat(),
        "checkpoint_in": cfg.checkpoint_in,
        "checkpoint_out": cfg.checkpoint_out,
        "baseline": {"ssim": round(baseline_ssim, 4), "latency_ms": round(baseline_latency, 2)},
        "result":   {"ssim": round(result_ssim, 4),   "latency_ms": round(result_latency, 2)},
        "speedup":   round(baseline_latency / result_latency, 3),
        "ssim_drop": round(baseline_ssim - result_ssim, 4),
        "groups": cfg.groups,
        "target": cfg.target,
        "finetuned": cfg.finetune.enabled,
        "finetune_epochs": cfg.finetune.epochs if cfg.finetune.enabled else 0,
    }
    path = save_results(results, to_absolute_path(cfg.results_dir), cfg.results_name)
    log.info(f"Results: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-batch quantization benchmark: FP32 / FP16 / LSQ-INT8 (FP16 inference).

Usage:
  python scripts/benchmark_quant_batches.py
  python scripts/benchmark_quant_batches.py benchmark.batch_sizes='[1,4,16,64]'
  python scripts/benchmark_quant_batches.py +benchmark.profile=true
  python scripts/benchmark_quant_batches.py +benchmark.e2e=false
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import time
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from src.acceleration.benchmark import measure_ssim, save_results
from src.acceleration.lsq import count_int8_model_size_mb
from src.acceleration.quantize import apply_lsq_ptq, count_model_size_mb
from src.datasets.data_utils import get_dataloaders
from src.utils import select_device, set_random_seed

log = logging.getLogger(__name__)


def measure_latency_detailed(model, device, input_size, n_warmup=20, n_runs=200):
    model.eval()
    input_size = tuple(int(x) for x in input_size)
    dtype = next(model.parameters()).dtype
    dummy = torch.randn(*input_size, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        torch.cuda.synchronize()

        timings = []
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

    arr = np.array(timings)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "std_ms": round(float(arr.std()), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def profile_model(model, device, input_size, trace_dir, tag, n_warmup=5, n_active=10):
    from torch.profiler import ProfilerActivity, profile, schedule

    input_size = tuple(int(x) for x in input_size)
    dtype = next(model.parameters()).dtype
    dummy = torch.randn(*input_size, device=device, dtype=dtype)
    model.eval()

    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    sched = schedule(wait=1, warmup=n_warmup, active=n_active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched, record_shapes=True, profile_memory=True, with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(1 + n_warmup + n_active):
                model(dummy)
                prof.step()

    chrome_trace = trace_path / f"{tag}_chrome_trace.json"
    prof.export_chrome_trace(str(chrome_trace))

    key_averages = prof.key_averages()

    def _ct(evt):
        for a in ("cuda_time_total", "self_cuda_time_total"):
            if hasattr(evt, a):
                return getattr(evt, a)
        return 0

    top_kernels = []
    for evt in sorted(key_averages, key=_ct, reverse=True)[:10]:
        ct = _ct(evt)
        if ct > 0:
            top_kernels.append({
                "name": evt.key,
                "cuda_time_total_us": round(ct, 1),
                "cuda_time_avg_us": round(ct / max(evt.count, 1), 1),
                "calls": evt.count,
            })

    table_path = trace_path / f"{tag}_profiler_table.txt"
    try:
        table_path.write_text(key_averages.table(sort_by="cuda_time_total", row_limit=30))
    except Exception:
        table_path.write_text(key_averages.table(sort_by="self_cuda_time_total", row_limit=30))

    return {"chrome_trace": str(chrome_trace), "top_cuda_kernels": top_kernels}


def measure_e2e_throughput(model, dataloader, device, max_batches=30, n_warmup_batches=5):
    model.eval()
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_warmup_batches:
                break
            model(batch["lr"].to(device=device, dtype=dtype)).clamp(0.0, 1.0)
    torch.cuda.synchronize()

    total_images = 0
    batches_done = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            lr = batch["lr"].to(device=device, dtype=dtype)
            model(lr).clamp(0.0, 1.0)
            total_images += lr.size(0)
            batches_done += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "total_images": total_images,
        "total_time_s": round(elapsed, 3),
        "throughput_ips": round(total_images / max(elapsed, 1e-9), 1),
        "avg_batch_ms": round((elapsed / max(batches_done, 1)) * 1000, 2),
        "batches_processed": batches_done,
    }


def _gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": "N/A"}
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": p.name, "gpu_memory_gb": round(p.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda or "N/A", "pytorch_version": torch.__version__,
    }


def _load_fp32(cfg, ckpt, device):
    m = instantiate(cfg.model).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def _make_fp16(cfg, ckpt, device):
    return _load_fp32(cfg, ckpt, device).half()


def _make_lsq_fp32(cfg, ckpt, device, val_loader):
    m = _load_fp32(cfg, ckpt, device)
    m, r = apply_lsq_ptq(m, val_loader, device,
                          n_bits_w=cfg.get("n_bits_w", 8),
                          skip_first_last=cfg.get("skip_first_last", True))
    return m, r


def _make_lsq_fp16(cfg, ckpt, device, val_loader):
    m, r = _make_lsq_fp32(cfg, ckpt, device, val_loader)
    m.half()
    return m, r


@hydra.main(version_base=None, config_path="../src/configs", config_name="quantize_ptq")
def main(cfg: DictConfig) -> None:
    set_random_seed(42)
    device = select_device("auto")

    batch_sizes = [int(b) for b in cfg.benchmark.get("batch_sizes", [1, 4, 16, 64])]
    C, H, W = 3, int(cfg.benchmark.get("height", 256)), int(cfg.benchmark.get("width", 256))
    n_warmup = int(cfg.benchmark.n_warmup)
    n_runs = int(cfg.benchmark.n_runs)
    do_profile = bool(cfg.benchmark.get("profile", False))
    do_e2e = bool(cfg.benchmark.get("e2e", True))
    e2e_max_batches = int(cfg.benchmark.get("e2e_max_batches", 30))
    results_dir = to_absolute_path(cfg.results_dir)
    trace_dir = str(Path(results_dir) / "profiler_traces")

    configs_order = ["fp32", "fp16", "lsq_int8_fp32", "lsq_int8_fp16"]
    gpu_info = _gpu_info()
    log.info(f"Device: {device} — {gpu_info.get('gpu_name', 'N/A')}")
    log.info(f"Batches: {batch_sizes}, {H}×{W}, warmup={n_warmup}, runs={n_runs}")

    ckpt = torch.load(to_absolute_path(cfg.checkpoint_in), map_location=device, weights_only=False)
    dataloaders, _ = get_dataloaders(cfg, device)
    val_loader = dataloaders.get("val") or dataloaders.get("test")

    # SSIM (once per config)
    log.info("Measuring SSIM...")
    ssim_results = {}

    m = _load_fp32(cfg, ckpt, device)
    fp32_size = count_model_size_mb(m)
    ssim_results["fp32"] = round(measure_ssim(m, val_loader, device), 4)
    log.info(f"  FP32  SSIM={ssim_results['fp32']:.4f}  size={fp32_size:.1f}MB")
    del m; torch.cuda.empty_cache()

    m = _make_fp16(cfg, ckpt, device)
    fp16_size = round(fp32_size / 2, 2)
    ssim_results["fp16"] = round(measure_ssim(m, val_loader, device), 4)
    log.info(f"  FP16  SSIM={ssim_results['fp16']:.4f}  size={fp16_size:.1f}MB")
    del m; torch.cuda.empty_cache()

    m, int8_result = _make_lsq_fp32(cfg, ckpt, device, val_loader)
    int8_size = round(count_int8_model_size_mb(int8_result), 2)
    ssim_results["lsq_int8_fp32"] = round(measure_ssim(m, val_loader, device), 4)
    log.info(f"  LSQ-FP32  SSIM={ssim_results['lsq_int8_fp32']:.4f}  size={int8_size:.1f}MB")
    del m; torch.cuda.empty_cache()

    m, _ = _make_lsq_fp16(cfg, ckpt, device, val_loader)
    ssim_results["lsq_int8_fp16"] = round(measure_ssim(m, val_loader, device), 4)
    log.info(f"  LSQ-FP16  SSIM={ssim_results['lsq_int8_fp16']:.4f}  size={int8_size:.1f}MB")
    del m; torch.cuda.empty_cache()

    sizes = {
        "fp32": round(fp32_size, 2), "fp16": fp16_size,
        "lsq_int8_fp32": int8_size, "lsq_int8_fp16": int8_size,
    }

    # Sweep batch sizes
    batch_results = {c: {} for c in configs_order}
    profile_results = {}

    for bs in batch_sizes:
        inp = (bs, C, H, W)
        log.info(f"--- batch={bs} ---")

        for cfg_name, builder in [
            ("fp32", lambda: _load_fp32(cfg, ckpt, device)),
            ("fp16", lambda: _make_fp16(cfg, ckpt, device)),
            ("lsq_int8_fp32", lambda: _make_lsq_fp32(cfg, ckpt, device, val_loader)[0]),
            ("lsq_int8_fp16", lambda: _make_lsq_fp16(cfg, ckpt, device, val_loader)[0]),
        ]:
            log.info(f"  [{cfg_name}]")
            model = builder()
            lat = measure_latency_detailed(model, device, inp, n_warmup, n_runs)
            lat["throughput_ips"] = round(bs / (lat["mean_ms"] / 1000), 1)
            batch_results[cfg_name][str(bs)] = lat
            log.info(f"    mean={lat['mean_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms "
                     f"tput={lat['throughput_ips']:.0f}/s")

            if do_profile:
                tag = f"{cfg_name}_b{bs}"
                profile_results.setdefault(cfg_name, {})[str(bs)] = \
                    profile_model(model, device, inp, trace_dir, tag)

            del model; torch.cuda.empty_cache()

    # End-to-end throughput
    e2e_results = {}
    if do_e2e and val_loader is not None:
        log.info("End-to-end throughput...")
        for cfg_name, builder in [
            ("fp32", lambda: _load_fp32(cfg, ckpt, device)),
            ("fp16", lambda: _make_fp16(cfg, ckpt, device)),
            ("lsq_int8_fp32", lambda: _make_lsq_fp32(cfg, ckpt, device, val_loader)[0]),
            ("lsq_int8_fp16", lambda: _make_lsq_fp16(cfg, ckpt, device, val_loader)[0]),
        ]:
            model = builder()
            e2e_results[cfg_name] = measure_e2e_throughput(model, val_loader, device, e2e_max_batches)
            log.info(f"  {cfg_name}: {e2e_results[cfg_name]['throughput_ips']:.1f} img/s")
            del model; torch.cuda.empty_cache()

        fp16_tput = e2e_results["fp16"]["throughput_ips"]
        for name, data in e2e_results.items():
            data["speedup_vs_fp16"] = round(data["throughput_ips"] / max(fp16_tput, 1e-6), 3)

    # Summary
    log.info("\n" + "=" * 80)
    log.info(f"SUMMARY — {gpu_info.get('gpu_name','N/A')}, {H}×{W}, runs={n_runs}")
    bs_hdr = "".join(f"{'B=' + str(b):>12}" for b in batch_sizes)
    log.info(f"{'Config':<12} {'SSIM':>7} {'MB':>6} {bs_hdr}")
    log.info("-" * (27 + 12 * len(batch_sizes)))
    for c in configs_order:
        lats = "".join(f"{batch_results[c][str(b)]['mean_ms']:>12.1f}" for b in batch_sizes)
        log.info(f"{c:<12} {ssim_results[c]:>7.4f} {sizes[c]:>6.1f} {lats}")

    # Speedup
    for c in configs_order:
        for b in batch_sizes:
            fp16_ms = batch_results["fp16"][str(b)]["mean_ms"]
            cfg_ms = batch_results[c][str(b)]["mean_ms"]
            batch_results[c][str(b)]["speedup_vs_fp16"] = round(fp16_ms / max(cfg_ms, 1e-6), 3)

    # Save JSON
    final = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "params": {"batch_sizes": batch_sizes, "H": H, "W": W, "n_warmup": n_warmup, "n_runs": n_runs},
        "ssim": ssim_results,
        "model_sizes_mb": sizes,
        "latency": batch_results,
    }
    if e2e_results:
        final["e2e_throughput"] = e2e_results
    if profile_results:
        final["profiling"] = profile_results

    out_path = save_results(final, results_dir, "quant_batch_sweep")
    log.info(f"Results: {out_path}")

    # Flat table for charts
    flat = []
    for c in configs_order:
        e2e = e2e_results.get(c, {})
        for b in batch_sizes:
            d = batch_results[c][str(b)]
            flat.append({
                "config": c, "batch_size": b,
                "ssim": ssim_results[c], "model_size_mb": sizes[c],
                "mean_ms": d["mean_ms"], "median_ms": d["median_ms"],
                "p95_ms": d["p95_ms"], "throughput_ips": d["throughput_ips"],
                "speedup_vs_fp16": d["speedup_vs_fp16"],
                "e2e_throughput_ips": e2e.get("throughput_ips"),
                "e2e_speedup_vs_fp16": e2e.get("speedup_vs_fp16"),
            })

    flat_path = Path(results_dir) / "quant_batch_sweep_flat.json"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flat_path, "w") as f:
        json.dump(flat, f, indent=2)
    log.info(f"Flat JSON: {flat_path}")


if __name__ == "__main__":
    main()

"""
Benchmark UNetSR x4 via Apache TVM (Relax frontend, CPU/LLVM, no autotuning).

TVM 0.24+ uses `relax` (the relay API was removed).  The model is converted
via torch.export -> tvm.relax.frontend.torch.from_exported_program, compiled
with the default LLVM pipeline and run through VirtualMachine.

Usage (from efml-unet/):
  python scripts/benchmark_tvm.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling import build_model, prepare_model_for_inference
from src.data import build_dataset
from src.metrics import compute_psnr, compute_ssim
from src.config import load_config

CROP_LR = 256
SCALE   = 4
WARMUP  = 1
MEASURE = 3
CONFIG  = "configs/sr_baseline_x4.yaml"


def build_loader(config: dict) -> DataLoader:
    crop_size = config["data"]["eval_crop_size"]
    dataset = build_dataset(
        lr_dir=config["data"]["val_lr_dir"],
        hr_dir=config["data"]["val_hr_dir"],
        scale=SCALE,
        crop_size=crop_size,
        training=False,
    )
    dataset.samples = [
        (lr, hr, n) for lr, hr, n in dataset.samples
        if min(Image.open(lr).size) >= crop_size
    ]
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


def compile_tvm_cpu(model: torch.nn.Module):
    """Export model via torch.export -> TVM Relax, compile for CPU/LLVM."""
    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    example_input = torch.randn(1, 3, CROP_LR, CROP_LR)
    model_cpu = model.cpu().eval()

    print("Exporting model with torch.export…")
    with torch.no_grad():
        exported = torch.export.export(model_cpu, (example_input,))

    print("Converting to TVM Relax IR…")
    # keep_params_as_input=False bakes weights into the IR as constants
    mod = from_exported_program(exported, keep_params_as_input=False, unwrap_unit_return_tuple=True)

    target = tvm.target.Target("llvm")
    dev    = tvm.cpu(0)

    print("Compiling with default LLVM pipeline (no autotuning)…")
    t0 = time.perf_counter()
    pipeline = relax.get_default_pipeline(target)
    mod = pipeline(mod)
    ex  = relax.build(mod, target=target)
    print(f"  Compilation done in {time.perf_counter() - t0:.1f}s")

    vm = relax.VirtualMachine(ex, dev)
    return vm, dev


def bench_tvm(vm, dev, loader: DataLoader) -> dict:
    import tvm

    latencies, psnrs, ssims = [], [], []

    for step, (lr, hr, _) in enumerate(loader):
        # Use DLPack zero-copy bridge from torch → TVM
        lr_tvm = tvm.ffi.from_dlpack(lr.contiguous())

        dev.sync()
        t0 = time.perf_counter()
        out = vm["main"](lr_tvm)
        dev.sync()
        elapsed = (time.perf_counter() - t0) * 1000

        pred_np = out.numpy()

        if step >= WARMUP:
            p = torch.from_numpy(pred_np[0]).clamp(0, 1)
            h = hr[0]
            psnrs.append(compute_psnr(p, h))
            ssims.append(compute_ssim(p, h))
            latencies.append(elapsed)
        if len(latencies) >= MEASURE:
            break

    return _stats("TVM Relax CPU (LLVM, no tuning)", latencies, psnrs, ssims)


def bench_eager_cpu(model: torch.nn.Module, loader: DataLoader) -> dict:
    device = torch.device("cpu")
    model  = prepare_model_for_inference(model, device, use_fp16=False)
    latencies, psnrs, ssims = [], [], []

    with torch.inference_mode():
        for step, (lr, hr, _) in enumerate(loader):
            lr = lr.to(device)
            t0 = time.perf_counter()
            pred = model(lr)
            elapsed = (time.perf_counter() - t0) * 1000

            if step >= WARMUP:
                p = pred[0].float().clamp(0, 1)
                h = hr[0].float()
                psnrs.append(compute_psnr(p, h))
                ssims.append(compute_ssim(p, h))
                latencies.append(elapsed)
            if len(latencies) >= MEASURE:
                break

    return _stats("Eager PyTorch CPU", latencies, psnrs, ssims)


def _stats(label: str, latencies: list, psnrs: list, ssims: list) -> dict:
    total_s = sum(latencies) / 1000
    return {
        "label": label,
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_std_ms":   float(np.std(latencies)),
        "latency_p50_ms":   float(np.percentile(latencies, 50)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        "throughput_img_s": float(len(latencies) / total_s),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def print_table(results: list[dict]) -> None:
    print(f"\n{'Method':<38} {'lat_mean':>10} {'lat_p95':>9} {'tput':>10} {'PSNR':>7} {'SSIM':>6}")
    print("-" * 84)
    baseline = results[0]["latency_mean_ms"]
    for r in results:
        speedup = baseline / r["latency_mean_ms"]
        print(
            f"{r['label']:<38} "
            f"{r['latency_mean_ms']:>8.1f}ms "
            f"{r['latency_p95_ms']:>8.1f}ms "
            f"{r['throughput_img_s']:>8.1f}/s "
            f"{r['psnr']:>7.3f} "
            f"{r['ssim']:>6.4f}"
            f"  (×{speedup:.2f})"
        )


if __name__ == "__main__":
    config = load_config(CONFIG)

    model = build_model(3, 3, 64, SCALE)
    ckpt  = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_loader(config)

    print("── Eager PyTorch CPU baseline ─────────────────────────────")
    r_eager = bench_eager_cpu(model, loader)
    print(json.dumps(r_eager, indent=2))

    print("\n── TVM Relax CPU (LLVM, no autotuning) ────────────────────")
    vm, dev = compile_tvm_cpu(model)
    r_tvm = bench_tvm(vm, dev, loader)
    print(json.dumps(r_tvm, indent=2))

    results = [r_eager, r_tvm]
    print_table(results)

    out = Path("results/tvm_benchmark_x4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

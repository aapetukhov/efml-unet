"""
Benchmark UNetSR x4 via Apache TVM (Relax ONNX frontend, CUDA).

Two compilation modes:
  --mode default  : ONNX → Relax → default pipeline (DLight Fallback, no tuning)
  --mode tuned    : ONNX → Relax → LegalizeOps → apply MetaSchedule database

The ONNX frontend is used because torch.export produces decomposed ops that
MetaSchedule cannot schedule; from_onnx retains high-level R.nn.conv2d ops.

Usage (from efml-unet/):
  python scripts/benchmark_tvm.py [--mode default|tuned]
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

CROP_LR   = 256
SCALE     = 4
WARMUP    = 10
MEASURE   = 50
CONFIG    = "configs/sr_baseline_x4.yaml"
ARCH      = "sm_86"
WORK_DIR  = "results/tvm_tuning"
ONNX_PATH = "results/unet_sr_x4.onnx"

# sm_86 (RTX 30xx / A10 / A30 / A40) hardware limits for MetaSchedule
_CUDA_TARGET = {
    "kind": "cuda",
    "arch": ARCH,
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
}


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


def load_onnx_relax():
    """Load ONNX model and convert to Relax IR with fully-static shapes (batch=1)."""
    import tvm
    import onnx as onnx_lib
    from tvm.relax.frontend.onnx import from_onnx
    onnx_model = onnx_lib.load(ONNX_PATH)
    mod = from_onnx(onnx_model, shape_dict={"input": [1, 3, CROP_LR, CROP_LR]},
                    keep_params_in_input=False)
    # Bind batch=1 — MetaSchedule cannot split symbolic loop bounds
    return tvm.relax.transform.BindSymbolicVars({"batch": 1})(mod)


def lower_mod(mod, target):
    """High-level Relax IR → lowered call_tir IR."""
    import tvm
    from tvm import relax
    return tvm.transform.Sequential([
        relax.transform.DecomposeOpsForInference(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])(mod)


def compile_tvm(model: torch.nn.Module, tuned: bool):
    """Return (vm, dev). Exports ONNX if needed, compiles for CUDA."""
    import tvm
    from tvm import relax
    from tvm.relax.backend.cuda.pipeline import dataflow_lower_passes, finalize_passes

    target = tvm.target.Target(_CUDA_TARGET)
    dev    = tvm.cuda(0)

    # Export ONNX if needed
    onnx_path = Path(ONNX_PATH)
    if not onnx_path.exists():
        print("Exporting model to ONNX…")
        dummy = torch.randn(1, 3, CROP_LR, CROP_LR)
        with torch.no_grad():
            torch.onnx.export(
                model.cpu().eval(), (dummy,), str(onnx_path),
                opset_version=17, input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )

    print("Loading ONNX → Relax IR…")
    mod = load_onnx_relax()

    if tuned:
        db_path = Path(WORK_DIR)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Tuning database not found at {WORK_DIR}. "
                "Run `python scripts/tune_tvm.py` first."
            )
        print(f"Applying MetaSchedule database from {WORK_DIR}…")
        t0 = time.perf_counter()
        from tvm.s_tir import meta_schedule as ms
        database = ms.database.JSONDatabase(work_dir=WORK_DIR)
        ex = ms.relax_integration.compile_relax(
            database=database,
            mod=mod,
            target=target,
            params=None,
        )
    else:
        print(f"Compiling with default CUDA pipeline ({ARCH}, no tuning)…")
        t0 = time.perf_counter()
        pipeline = relax.get_default_pipeline(target)
        mod = pipeline(mod)
        ex = relax.build(mod, target=target)

    print(f"  Compilation done in {time.perf_counter() - t0:.1f}s")

    vm = relax.VirtualMachine(ex, dev)
    return vm, dev


def bench_tvm_cuda(vm, dev, loader: DataLoader, label: str) -> dict:
    import tvm

    latencies, psnrs, ssims = [], [], []

    for step, (lr, hr, _) in enumerate(loader):
        lr_gpu = lr.cuda().contiguous()
        lr_tvm = tvm.ffi.from_dlpack(lr_gpu)

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

    return _stats(label, latencies, psnrs, ssims)


def bench_eager_cuda(model: torch.nn.Module, loader: DataLoader) -> dict:
    device = torch.device("cuda")
    model  = prepare_model_for_inference(model, device, use_fp16=False)
    latencies, psnrs, ssims = [], [], []

    with torch.inference_mode():
        for step, (lr, hr, _) in enumerate(loader):
            lr = lr.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(lr)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            if step >= WARMUP:
                p = pred[0].float().clamp(0, 1).cpu()
                h = hr[0].float()
                psnrs.append(compute_psnr(p, h))
                ssims.append(compute_ssim(p, h))
                latencies.append(elapsed)
            if len(latencies) >= MEASURE:
                break

    return _stats("Eager PyTorch CUDA FP32", latencies, psnrs, ssims)


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
    print(f"\n{'Method':<46} {'lat_mean':>10} {'lat_p95':>9} {'tput':>10} {'PSNR':>7} {'SSIM':>6}")
    print("-" * 92)
    baseline = results[0]["latency_mean_ms"]
    for r in results:
        speedup = baseline / r["latency_mean_ms"]
        print(
            f"{r['label']:<46} "
            f"{r['latency_mean_ms']:>8.1f}ms "
            f"{r['latency_p95_ms']:>8.1f}ms "
            f"{r['throughput_img_s']:>8.1f}/s "
            f"{r['psnr']:>7.3f} "
            f"{r['ssim']:>6.4f}"
            f"  (×{speedup:.2f})"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "tuned"], default="default",
                        help="default: compile fresh with DLight; tuned: apply MetaSchedule DB")
    cli = parser.parse_args()

    config = load_config(CONFIG)

    model = build_model(3, 3, 64, SCALE)
    ckpt  = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_loader(config)

    print("── Eager PyTorch CUDA FP32 baseline ───────────────────────")
    r_eager = bench_eager_cuda(model, loader)
    print(json.dumps(r_eager, indent=2))

    tuned = cli.mode == "tuned"
    label = f"TVM Relax CUDA {ARCH} ({'MetaSchedule tuned' if tuned else 'default pipeline'})"
    print(f"\n── {label} ─────────────────────────")
    vm, dev = compile_tvm(model, tuned=tuned)
    r_tvm = bench_tvm_cuda(vm, dev, loader, label)
    print(json.dumps(r_tvm, indent=2))

    results = [r_eager, r_tvm]
    print_table(results)

    suffix = "tuned" if tuned else "default"
    out = Path(f"results/tvm_benchmark_x4_{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

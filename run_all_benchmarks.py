"""
run_all_benchmarks.py — portable benchmark runner for UNetSR variants.

By default uses random noise as input (no dataset needed).
Pass --use-real-data to measure PSNR/SSIM on actual images.

Runs all available backends and writes a single consolidated JSON.
Missing backends (TensorRT, TVM) are skipped gracefully.

Examples
--------
# Minimal — just needs the checkpoint:
python run_all_benchmarks.py --checkpoint unet_sr_x4_baseline.pt --scale 4

# With real data + FP16:
python run_all_benchmarks.py --checkpoint unet_sr_x4_baseline.pt --scale 4 \
    --use-real-data --fp16 \
    --val-lr-dir data/div2k/DIV2K_valid_LR_bicubic/X4 \
    --val-hr-dir data/div2k/DIV2K_valid_HR

# Different architecture:
python run_all_benchmarks.py --checkpoint my_model.pt --scale 8 --base-channels 32

# From YAML config (overrides arch/path args):
python run_all_benchmarks.py --config configs/sr_baseline_x4.yaml
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from src.modeling import build_model, prepare_model_for_inference
from src.metrics import compute_psnr, compute_ssim


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (overrides arch/path args)")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--out-channels", type=int, default=3)
    p.add_argument("--input-size", type=int, default=256, help="LR crop size (H=W)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    # Backends
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile runs")
    p.add_argument("--no-ort", action="store_true", help="Skip ORT runs")
    p.add_argument("--no-tvm", action="store_true", help="Skip TVM run")
    p.add_argument("--fp16", action="store_true", help="Also run FP16 eager")
    # Real data
    p.add_argument("--use-real-data", action="store_true")
    p.add_argument("--val-lr-dir", default=None)
    p.add_argument("--val-hr-dir", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    # Numerical check tolerance (set 0 to skip)
    p.add_argument("--check-atol", type=float, default=1e-3,
                   help="Max abs diff vs eager for correctness check (0 = skip)")
    # Output
    p.add_argument("--output", default=None, help="JSON output path (auto-named if omitted)")
    return p.parse_args()


# ── Loaders ───────────────────────────────────────────────────────────────────

def make_noise_loader(in_ch: int, lr_size: int, scale: int,
                      batch_size: int, warmup: int, steps: int) -> DataLoader:
    n = (warmup + steps) * batch_size
    lr = torch.rand(n, in_ch, lr_size, lr_size)
    hr = torch.rand(n, in_ch, lr_size * scale, lr_size * scale)
    return DataLoader(TensorDataset(lr, hr), batch_size=batch_size, shuffle=False)


def make_real_loader(lr_dir: str, hr_dir: str, scale: int, crop_size: int,
                     batch_size: int, num_workers: int) -> DataLoader:
    from PIL import Image
    from src.data import build_dataset

    dataset = build_dataset(
        lr_dir=lr_dir, hr_dir=hr_dir, scale=scale, crop_size=crop_size, training=False
    )
    dataset.samples = [
        (lr, hr, n) for lr, hr, n in dataset.samples
        if min(Image.open(lr).size) >= crop_size
    ]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def loader_iter(loader: DataLoader):
    """Yield (lr, hr) regardless of whether loader has 2 or 3 columns."""
    for batch in loader:
        yield batch[0], batch[1]


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _bench_gpu(
    run_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    steps: int,
    use_fp16: bool,
    label: str,
    compute_quality: bool,
    batch_size: int,
) -> dict:
    """GPU benchmark using CUDA Events for accurate per-kernel timing."""
    latencies, psnrs, ssims = [], [], []
    measured = 0
    peak_mem_mb = 0.0

    torch.cuda.reset_peak_memory_stats(device)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    for step, (lr, hr) in enumerate(loader_iter(loader)):
        lr = lr.to(device, non_blocking=True)
        if use_fp16:
            lr = lr.half()

        start_ev.record()
        pred = run_fn(lr)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed = start_ev.elapsed_time(end_ev)  # ms

        if step >= warmup:
            latencies.append(elapsed)
            total_imgs_so_far = len(latencies) * batch_size
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            if compute_quality:
                bs = lr.size(0)
                for i in range(bs):
                    p = pred[i].float().clamp(0, 1).cpu()
                    h = hr[i].float()
                    psnrs.append(compute_psnr(p, h))
                    ssims.append(compute_ssim(p, h))
            measured += 1

        if measured >= steps:
            break

    total_s  = sum(latencies) / 1000
    total_imgs = measured * batch_size
    result = {
        "label": label,
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_std_ms":   float(np.std(latencies)),
        "latency_p50_ms":   float(np.percentile(latencies, 50)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        "throughput_img_s": float(total_imgs / total_s),
        "peak_mem_mb":      round(peak_mem_mb, 1),
    }
    if compute_quality and psnrs:
        result["psnr"] = float(np.mean(psnrs))
        result["ssim"] = float(np.mean(ssims))
    return result


def _bench_cpu(
    run_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    warmup: int,
    steps: int,
    label: str,
    compute_quality: bool,
    batch_size: int,
) -> dict:
    latencies, psnrs, ssims = [], [], []
    measured = 0

    for step, (lr, hr) in enumerate(loader_iter(loader)):
        t0 = time.perf_counter()
        pred = run_fn(lr)
        elapsed = (time.perf_counter() - t0) * 1000

        if step >= warmup:
            latencies.append(elapsed)
            if compute_quality:
                bs = lr.size(0)
                for i in range(bs):
                    p = pred[i].float().clamp(0, 1)
                    h = hr[i].float()
                    psnrs.append(compute_psnr(p, h))
                    ssims.append(compute_ssim(p, h))
            measured += 1

        if measured >= steps:
            break

    total_s = sum(latencies) / 1000
    total_imgs = measured * batch_size
    result = {
        "label": label,
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_std_ms":   float(np.std(latencies)),
        "latency_p50_ms":   float(np.percentile(latencies, 50)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        "throughput_img_s": float(total_imgs / total_s),
    }
    if compute_quality and psnrs:
        result["psnr"] = float(np.mean(psnrs))
        result["ssim"] = float(np.mean(ssims))
    return result


def _reset_gpu_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


# ── Per-backend runners ───────────────────────────────────────────────────────

def run_eager(model: nn.Module, loader, device, warmup, steps, batch_size,
              use_fp16, compute_quality) -> dict:
    dtype_str = "FP16" if use_fp16 else "FP32"
    label = f"Eager {dtype_str} ({device.type.upper()})"
    m = prepare_model_for_inference(copy.deepcopy(model), device, use_fp16=use_fp16)
    with torch.inference_mode():
        if device.type == "cuda":
            return _bench_gpu(m, loader, device, warmup, steps, use_fp16,
                               label, compute_quality, batch_size)
        return _bench_cpu(m, loader, warmup, steps, label, compute_quality, batch_size)


def run_compiled(model: nn.Module, loader, device, warmup, steps, batch_size,
                 mode, compute_quality) -> dict:
    label = f"torch.compile [{mode}] (CUDA)"
    # max-autotune needs extra warmup headroom for per-kernel autotuning
    effective_warmup = max(warmup, 20) if mode == "max-autotune" else warmup
    m = prepare_model_for_inference(copy.deepcopy(model), device, use_fp16=False)
    m = torch.compile(m, mode=mode)
    with torch.inference_mode():
        return _bench_gpu(m, loader, device, effective_warmup, steps, False,
                          label, compute_quality, batch_size)


def run_ort(onnx_path: Path, loader, device, provider: str,
            provider_options: dict | None, warmup, steps, batch_size,
            compute_quality) -> dict:
    import onnxruntime as ort

    providers = [(provider, provider_options or {}), "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
    active_provider = sess.get_providers()[0]
    if active_provider != provider:
        raise RuntimeError(f"{provider} unavailable, fell back to {active_provider}")
    active = active_provider.replace("ExecutionProvider", " EP")

    latencies, psnrs, ssims = [], [], []
    measured = 0
    peak_mem_mb = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, (lr, hr) in enumerate(loader_iter(loader)):
        lr_np = lr.numpy()

        t0 = time.perf_counter()
        pred_np = sess.run(["hr"], {"lr": lr_np})[0]
        elapsed = (time.perf_counter() - t0) * 1000

        if step >= warmup:
            latencies.append(elapsed)
            if device.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            if compute_quality:
                bs = lr.size(0)
                for i in range(bs):
                    p = torch.from_numpy(pred_np[i]).clamp(0, 1)
                    h = hr[i].float()
                    psnrs.append(compute_psnr(p, h))
                    ssims.append(compute_ssim(p, h))
            measured += 1

        if measured >= steps:
            break

    total_s = sum(latencies) / 1000
    total_imgs = measured * batch_size
    result = {
        "label": f"ORT {active}",
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_std_ms":   float(np.std(latencies)),
        "latency_p50_ms":   float(np.percentile(latencies, 50)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        "throughput_img_s": float(total_imgs / total_s),
    }
    if device.type == "cuda":
        result["peak_mem_mb"] = round(peak_mem_mb, 1)
    if compute_quality and psnrs:
        result["psnr"] = float(np.mean(psnrs))
        result["ssim"] = float(np.mean(ssims))
    return result


def run_tvm_cpu(model: nn.Module, loader, lr_size: int, in_ch: int,
                warmup, steps, batch_size, compute_quality) -> dict:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    model_cpu = copy.deepcopy(model).cpu().eval()
    example = torch.randn(1, in_ch, lr_size, lr_size)
    with torch.no_grad():
        ep = torch.export.export(model_cpu, (example,))

    mod = from_exported_program(ep, keep_params_as_input=False, unwrap_unit_return_tuple=True)
    target = tvm.target.Target("llvm")
    dev    = tvm.cpu(0)

    pipeline = relax.get_default_pipeline(target)
    mod = pipeline(mod)
    ex  = relax.build(mod, target=target)
    vm  = relax.VirtualMachine(ex, dev)

    latencies, psnrs, ssims = [], [], []
    measured = 0

    for step, (lr, hr) in enumerate(loader_iter(loader)):
        lr_tvm = tvm.ffi.from_dlpack(lr.contiguous())
        dev.sync()
        t0 = time.perf_counter()
        out = vm["main"](lr_tvm)
        dev.sync()
        elapsed = (time.perf_counter() - t0) * 1000

        if step >= warmup:
            latencies.append(elapsed)
            if compute_quality:
                pred_np = out.numpy()
                bs = lr.size(0)
                for i in range(bs):
                    p = torch.from_numpy(pred_np[i]).clamp(0, 1)
                    h = hr[i].float()
                    psnrs.append(compute_psnr(p, h))
                    ssims.append(compute_ssim(p, h))
            measured += 1

        if measured >= steps:
            break

    total_s = sum(latencies) / 1000
    total_imgs = measured * batch_size
    result = {
        "label": "TVM Relax CPU (LLVM, no tuning)",
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_std_ms":   float(np.std(latencies)),
        "latency_p50_ms":   float(np.percentile(latencies, 50)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        "throughput_img_s": float(total_imgs / total_s),
    }
    if compute_quality and psnrs:
        result["psnr"] = float(np.mean(psnrs))
        result["ssim"] = float(np.mean(ssims))
    return result


# ── Numerical correctness check ───────────────────────────────────────────────

def _check_output(label: str, eager_out: torch.Tensor, other_out: np.ndarray,
                  atol: float) -> None:
    diff = float(np.abs(eager_out.numpy() - other_out).max())
    status = "OK" if diff <= atol else f"WARN max|Δ|={diff:.2e} > tol={atol}"
    print(f"  Numerical check vs eager: {status}")


# ── Table ─────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    has_quality = any("psnr" in r for r in results)
    has_mem     = any("peak_mem_mb" in r for r in results)
    hdr = f"{'Method':<40} {'lat_mean':>10} {'lat_p95':>9} {'tput':>10}"
    if has_mem:
        hdr += f" {'mem_mb':>8}"
    if has_quality:
        hdr += f" {'PSNR':>7} {'SSIM':>6}"
    print(f"\n{hdr}")
    print("─" * (len(hdr) + 12))
    baseline = results[0]["latency_mean_ms"]
    for r in results:
        speedup = baseline / r["latency_mean_ms"]
        line = (
            f"{r['label']:<40} "
            f"{r['latency_mean_ms']:>8.1f}ms "
            f"{r['latency_p95_ms']:>8.1f}ms "
            f"{r['throughput_img_s']:>8.1f}/s"
            f"  (×{speedup:.2f})"
        )
        if has_mem:
            line += f"  {r.get('peak_mem_mb', 0):>6.0f}"
        if has_quality and "psnr" in r:
            line += f"  {r['psnr']:>7.3f}  {r['ssim']:>6.4f}"
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.config:
        from src.config import load_config
        cfg = load_config(args.config)
        args.checkpoint    = cfg["train"]["save_path"]
        args.scale         = cfg["data"]["scale"]
        args.base_channels = cfg["model"]["base_channels"]
        args.in_channels   = cfg["model"]["in_channels"]
        args.out_channels  = cfg["model"]["out_channels"]
        args.input_size    = cfg["data"]["eval_crop_size"]
        if args.use_real_data:
            args.val_lr_dir = cfg["data"]["val_lr_dir"]
            args.val_hr_dir = cfg["data"]["val_hr_dir"]

    if args.checkpoint is None:
        sys.exit("Error: --checkpoint required (or --config)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    # Collect env info for meta
    import platform
    cuda_version = torch.version.cuda or "N/A"
    cudnn_version = str(torch.backends.cudnn.version()) if device.type == "cuda" else "N/A"
    meta = {
        "timestamp":    datetime.now().isoformat(),
        "device":       str(device),
        "device_name":  device_name,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "platform":     platform.platform(),
        "scale":        args.scale,
        "input_size":   args.input_size,
        "base_channels": args.base_channels,
        "batch_size":   args.batch_size,
        "warmup":       args.warmup,
        "steps":        args.steps,
        "checkpoint":   str(args.checkpoint),
    }

    print(f"Device     : {device_name}")
    print(f"PyTorch    : {torch.__version__}  CUDA {cuda_version}  cuDNN {cudnn_version}")
    print(f"Scale      : {args.scale}x  |  Input: {args.input_size}²  |  base_ch: {args.base_channels}")
    print(f"Warmup     : {args.warmup}  |  Measure: {args.steps}  |  BS: {args.batch_size}")

    # Build model — auto-detect heavy checkpoint by presence of embedded config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embedded_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    is_heavy = (
        embedded_cfg is not None and
        "SRUNetHeavy" in str(embedded_cfg.get("model", {}).get("_target_", ""))
    )
    if is_heavy:
        from src.heavy_modeling import load_heavy_model
        model = load_heavy_model(args.checkpoint)
        cfg_model = embedded_cfg["model"]
        args.in_channels   = cfg_model.get("in_channels", 3)
        args.out_channels  = cfg_model.get("out_channels", 3)
        args.base_channels = cfg_model.get("base_channels", 96)
        args.scale         = embedded_cfg.get("datasets", {}).get("scale", args.scale)
        print(f"Model      : SRUNetHeavy (base_ch={args.base_channels}, "
              f"{sum(p.numel() for p in model.parameters()):,} params)")
    else:
        model = build_model(args.in_channels, args.out_channels, args.base_channels, args.scale)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    model.eval()

    # ONNX path keyed by architecture so stale exports are never reused
    arch_hash = hashlib.md5(
        f"{args.in_channels},{args.out_channels},{args.base_channels},{args.scale},{args.input_size}".encode()
    ).hexdigest()[:8]
    onnx_path = Path(f"results/unet_sr_{arch_hash}.onnx")

    # Loaders
    compute_quality = False
    if args.use_real_data and args.val_lr_dir and args.val_hr_dir:
        loader = make_real_loader(args.val_lr_dir, args.val_hr_dir, args.scale,
                                  args.input_size, args.batch_size, args.num_workers)
        compute_quality = True
        print("Data       : real validation images (PSNR/SSIM enabled)")
    else:
        loader = make_noise_loader(args.in_channels, args.input_size, args.scale,
                                   args.batch_size, args.warmup, args.steps)
        print("Data       : random noise (latency only)")

    results: list[dict] = []
    eager_ref: torch.Tensor | None = None  # for numerical checks

    def try_run(name: str, fn: Callable) -> None:
        nonlocal eager_ref
        print(f"\n── {name} {'─' * max(0, 55 - len(name))}")
        _reset_gpu_state(device)
        try:
            r = fn()
            results.append(r)
            mem = f"  {r['peak_mem_mb']:.0f} MB" if "peak_mem_mb" in r else ""
            print(f"  {r['label']}: {r['latency_mean_ms']:.1f} ms mean, "
                  f"{r['throughput_img_s']:.1f} img/s{mem}")
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── 1. Eager FP32 (reference) ─────────────────────────────────────────────
    try_run("Eager FP32", lambda: run_eager(
        model, loader, device, args.warmup, args.steps, args.batch_size,
        False, compute_quality))

    # Capture one eager output for downstream correctness checks
    if args.check_atol > 0 and device.type == "cuda":
        _m = prepare_model_for_inference(copy.deepcopy(model), device, use_fp16=False)
        _x = next(loader_iter(loader))[0].to(device)
        with torch.inference_mode():
            eager_ref = _m(_x).float().cpu()
        del _m

    # ── 2. Eager FP16 ────────────────────────────────────────────────────────
    if args.fp16 and device.type == "cuda":
        try_run("Eager FP16", lambda: run_eager(
            model, loader, device, args.warmup, args.steps, args.batch_size,
            True, compute_quality))

    # ── 3. torch.compile ─────────────────────────────────────────────────────
    if not args.no_compile and device.type == "cuda":
        for mode in ("default", "reduce-overhead", "max-autotune"):
            try_run(f"torch.compile [{mode}]", lambda m=mode: run_compiled(
                model, loader, device, args.warmup, args.steps,
                args.batch_size, m, compute_quality))

    # ── 4. ORT ───────────────────────────────────────────────────────────────
    if not args.no_ort:
        if importlib.util.find_spec("onnxruntime") is None:
            print("\n── ORT ── SKIPPED: onnxruntime not installed")
        else:
            import onnxruntime as ort

            if not onnx_path.exists():
                print(f"\nExporting ONNX model → {onnx_path} …")
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                x_dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)
                torch.onnx.export(
                    copy.deepcopy(model).cpu().eval(), x_dummy, str(onnx_path),
                    opset_version=17,
                    input_names=["lr"], output_names=["hr"],
                    dynamic_axes={"lr": {0: "batch"}, "hr": {0: "batch"}},
                )

            # Correctness check for ORT
            if args.check_atol > 0 and eager_ref is not None:
                import onnxruntime as _ort
                _sess = _ort.InferenceSession(str(onnx_path),
                                              providers=["CPUExecutionProvider"])
                _out = _sess.run(["hr"], {"lr": _x.cpu().numpy()})[0]
                _check_output("ORT CPU", eager_ref, _out, args.check_atol)

            available = ort.get_available_providers()

            if "CUDAExecutionProvider" in available:
                try_run("ORT CUDA EP", lambda: run_ort(
                    onnx_path, loader, device, "CUDAExecutionProvider", None,
                    args.warmup, args.steps, args.batch_size, compute_quality))

            if "TensorrtExecutionProvider" in available:
                trt_cache = "results/trt_cache"
                trt_base = {"trt_engine_cache_enable": True,
                            "trt_engine_cache_path": trt_cache}
                try_run("ORT TensorRT FP32", lambda: run_ort(
                    onnx_path, loader, device, "TensorrtExecutionProvider",
                    {**trt_base, "trt_fp16_enable": False},
                    args.warmup, args.steps, args.batch_size, compute_quality))
                try_run("ORT TensorRT FP16", lambda: run_ort(
                    onnx_path, loader, device, "TensorrtExecutionProvider",
                    {**trt_base, "trt_fp16_enable": True},
                    args.warmup, args.steps, args.batch_size, compute_quality))

    # ── 5. TVM ───────────────────────────────────────────────────────────────
    if not args.no_tvm:
        if importlib.util.find_spec("tvm") is None:
            print("\n── TVM ── SKIPPED: tvm not installed")
        else:
            try_run("TVM Relax CPU (LLVM)", lambda: run_tvm_cpu(
                model, loader, args.input_size, args.in_channels,
                args.warmup, args.steps, args.batch_size, compute_quality))

    # ── Summary ───────────────────────────────────────────────────────────────
    print_table(results)

    out_path = (Path(args.output) if args.output else
                Path(f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

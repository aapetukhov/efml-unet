"""
Benchmark UNetSR x4 via ONNX Runtime:
  - CUDAExecutionProvider
  - TensorrtExecutionProvider (if available)

Compares against eager PyTorch baseline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling import build_model, prepare_model_for_inference
from src.data import build_dataset
from src.metrics import compute_psnr, compute_ssim
from src.config import load_config

CROP_LR   = 256
SCALE     = 4
WARMUP    = 10
MEASURE   = 50
ONNX_PATH = Path("results/unet_sr_x4.onnx")
CONFIG    = "configs/sr_baseline_x4.yaml"


# ── Export to ONNX ───────────────────────────────────────────────────────────
def export_onnx(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, CROP_LR, CROP_LR)
    torch.onnx.export(
        model.cpu(),
        x,
        str(path),
        opset_version=17,
        input_names=["lr"],
        output_names=["hr"],
        dynamic_axes={"lr": {0: "batch"}, "hr": {0: "batch"}},
    )
    print(f"Exported ONNX → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ── Build dataloader ─────────────────────────────────────────────────────────
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


# ── Eager PyTorch baseline ───────────────────────────────────────────────────
def bench_eager(model: torch.nn.Module, loader: DataLoader) -> dict:
    device = torch.device("cuda")
    model = prepare_model_for_inference(model, device, use_fp16=False)
    latencies, psnrs, ssims = [], [], []

    with torch.inference_mode():
        for step, (lr, hr, _) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(lr)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            if step >= WARMUP:
                p = pred[0].float().clamp(0, 1).cpu()
                h = hr[0].float().cpu()
                psnrs.append(compute_psnr(p, h))
                ssims.append(compute_ssim(p, h))
                latencies.append(elapsed)
            if len(latencies) >= MEASURE:
                break

    return _stats("eager (FP32)", latencies, psnrs, ssims)


# ── ORT benchmark ────────────────────────────────────────────────────────────
def bench_ort(loader: DataLoader, provider: str, provider_options: dict | None = None) -> dict:
    import onnxruntime as ort

    providers = [(provider, provider_options or {}), "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_PATH), sess_opts, providers=providers)

    # Check which provider was actually used
    active = sess.get_providers()[0]

    latencies, psnrs, ssims = [], [], []

    for step, (lr, hr, _) in enumerate(loader):
        lr_np = lr.numpy()
        hr_np = hr.numpy()

        t0 = time.perf_counter()
        pred_np = sess.run(["hr"], {"lr": lr_np})[0]
        elapsed = (time.perf_counter() - t0) * 1000

        if step >= WARMUP:
            p = torch.from_numpy(pred_np[0]).clamp(0, 1)
            h = torch.from_numpy(hr_np[0])
            psnrs.append(compute_psnr(p, h))
            ssims.append(compute_ssim(p, h))
            latencies.append(elapsed)
        if len(latencies) >= MEASURE:
            break

    label = f"ORT {active.replace('ExecutionProvider', ' EP')}"
    return _stats(label, latencies, psnrs, ssims)


def _stats(label: str, latencies: list, psnrs: list, ssims: list) -> dict:
    total_s = sum(latencies) / 1000
    return {
        "label": label,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms":  float(np.std(latencies)),
        "latency_p50_ms":  float(np.percentile(latencies, 50)),
        "latency_p95_ms":  float(np.percentile(latencies, 95)),
        "throughput_img_s": float(len(latencies) / total_s),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def print_table(results: list[dict]) -> None:
    print(f"\n{'Method':<30} {'lat_mean':>10} {'lat_p95':>9} {'tput':>10} {'PSNR':>7} {'SSIM':>6}")
    print("-" * 76)
    baseline = results[0]["latency_mean_ms"]
    for r in results:
        speedup = baseline / r["latency_mean_ms"]
        print(
            f"{r['label']:<30} "
            f"{r['latency_mean_ms']:>8.1f}ms "
            f"{r['latency_p95_ms']:>8.1f}ms "
            f"{r['throughput_img_s']:>8.1f}/s "
            f"{r['psnr']:>7.3f} "
            f"{r['ssim']:>6.4f}"
            f"  (×{speedup:.2f})"
        )


if __name__ == "__main__":
    import onnxruntime as ort

    config = load_config(CONFIG)

    # Load model
    model = build_model(3, 3, 64, SCALE)
    ckpt = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Export once
    if not ONNX_PATH.exists():
        export_onnx(model, ONNX_PATH)
    else:
        print(f"Using cached ONNX: {ONNX_PATH}")

    loader = build_loader(config)
    available = ort.get_available_providers()
    results = []

    print("\n── Eager PyTorch baseline ──────────────────────")
    results.append(bench_eager(model, loader))

    print("\n── ORT CUDA EP ─────────────────────────────────")
    results.append(bench_ort(loader, "CUDAExecutionProvider"))

    if "TensorrtExecutionProvider" in available:
        print("\n── ORT TensorRT EP ─────────────────────────────")
        trt_opts = {
            "trt_fp16_enable": False,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "results/trt_cache",
        }
        try:
            results.append(bench_ort(loader, "TensorrtExecutionProvider", trt_opts))
        except Exception as e:
            print(f"TRT EP failed: {e}")

        print("\n── ORT TensorRT EP (FP16) ──────────────────────")
        trt_opts_fp16 = {**trt_opts, "trt_fp16_enable": True}
        try:
            results.append(bench_ort(loader, "TensorrtExecutionProvider", trt_opts_fp16))
        except Exception as e:
            print(f"TRT EP FP16 failed: {e}")

    print_table(results)

    out = Path("results/ort_benchmark_x4.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

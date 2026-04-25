from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image

from PIL import Image

from src.config import load_config
from src.data import build_dataset
from src.metrics import compute_psnr, compute_ssim
from src.modeling import build_model, prepare_model_for_inference


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def benchmark_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    use_fp16: bool,
) -> Dict[str, float]:
    latencies_ms: List[float] = []
    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    measured_steps = 0
    total_images = 0

    with torch.inference_mode():
        for step, (lr_images, hr_images, _) in enumerate(dataloader):
            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)
            if use_fp16 and device.type == "cuda":
                lr_images = lr_images.half()

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = model(lr_images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            batch_size = lr_images.size(0)

            if step >= warmup_steps:
                for i in range(batch_size):
                    prediction = predictions[i].float().clamp(0.0, 1.0).cpu()
                    target = hr_images[i].float().cpu()
                    psnr_scores.append(compute_psnr(prediction, target))
                    ssim_scores.append(compute_ssim(prediction, target))

                latencies_ms.append(elapsed_ms)
                measured_steps += 1
                total_images += batch_size

            if measured_steps >= measure_steps:
                break

    if not latencies_ms:
        raise RuntimeError(
            "No measured steps collected. "
            "Reduce warmup_steps or increase dataset size."
        )

    total_time_s = sum(latencies_ms) / 1000.0
    throughput = total_images / total_time_s if total_time_s > 0 else 0.0

    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_std_ms": float(np.std(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_images_per_s": float(throughput),
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "measured_steps": measured_steps,
        "total_images": total_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to sweep. Overrides config batch_size.",
    )
    parser.add_argument(
        "--compile",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
        help="Enable torch.compile with the given mode.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup_steps from config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["benchmark"]["device"])
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu"
    )
    use_fp16 = bool(config["benchmark"]["use_fp16"])

    crop_size = config["data"]["eval_crop_size"]
    dataset = build_dataset(
        lr_dir=config["data"]["val_lr_dir"],
        hr_dir=config["data"]["val_hr_dir"],
        scale=config["data"]["scale"],
        crop_size=crop_size,
        training=False,
    )
    # Drop images whose LR shorter side is smaller than crop_size — they
    # would produce differently-sized tensors that break batched collation.
    before = len(dataset)
    dataset.samples = [
        (lr, hr, name)
        for lr, hr, name in dataset.samples
        if min(Image.open(lr).size) >= crop_size
    ]
    if len(dataset) < before:
        print(f"Filtered {before - len(dataset)}/{before} images (LR shorter side < {crop_size}px).")

    model = build_model(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        scale=config["data"]["scale"],
    )
    checkpoint = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = prepare_model_for_inference(model=model, device=device, use_fp16=use_fp16)

    if args.compile is not None:
        print(f"Compiling model with mode='{args.compile}' (first warmup step will be slow)...")
        model = torch.compile(model, mode=args.compile)

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else config["benchmark"]["warmup_steps"]
    batch_sizes = args.batch_sizes or [config["benchmark"].get("batch_size", 1)]
    all_results = []

    for bs in batch_sizes:
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=device.type == "cuda",
        )
        try:
            metrics = benchmark_model(
                model=model,
                dataloader=dataloader,
                device=device,
                warmup_steps=warmup_steps,
                measure_steps=config["benchmark"]["measure_steps"],
                use_fp16=use_fp16,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n--- batch_size={bs} --- OOM, skipping")
            continue
        metrics.update(
            {
                "experiment_name": config["experiment_name"],
                "device": str(device),
                "device_name": device_name,
                "use_fp16": use_fp16,
                "compile_mode": args.compile,
                "scale": config["data"]["scale"],
                "batch_size": bs,
            }
        )
        all_results.append(metrics)
        print(f"\n--- batch_size={bs} ---")
        print(json.dumps(metrics, indent=2))

    output_path = Path(config["output"]["benchmark_metrics_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(all_results) == 1:
        result_to_save = all_results[0]
    else:
        result_to_save = all_results
        # Print summary table
        print("\n\n=== Batch Size Sweep Summary ===")
        print(f"{'BS':>4}  {'lat_mean':>10}  {'lat_p95':>9}  {'throughput':>14}  {'PSNR':>7}  {'SSIM':>6}")
        for r in all_results:
            print(
                f"{r['batch_size']:>4}  "
                f"{r['latency_mean_ms']:>9.1f}ms  "
                f"{r['latency_p95_ms']:>8.1f}ms  "
                f"{r['throughput_images_per_s']:>12.1f}/s  "
                f"{r['psnr']:>7.3f}  "
                f"{r['ssim']:>6.4f}"
            )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_to_save, file, indent=2)


if __name__ == "__main__":
    main()

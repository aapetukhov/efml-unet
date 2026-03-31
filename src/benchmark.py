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

            if step >= warmup_steps:
                predictions_cpu = predictions.float().cpu()
                targets_cpu = hr_images.float().cpu()
                for prediction, target in zip(predictions_cpu, targets_cpu):
                    psnr_scores.append(compute_psnr(prediction, target))
                    ssim_scores.append(compute_ssim(prediction, target))

                latencies_ms.append(elapsed_ms)
                measured_steps += 1
                total_images += lr_images.shape[0]

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
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["benchmark"]["device"])

    dataset = build_dataset(
        hr_dir=config["data"]["val_hr_dir"],
        scale=config["data"]["scale"],
        crop_size=config["data"]["eval_crop_size"],
        training=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    )
    checkpoint = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = prepare_model_for_inference(
        model=model,
        device=device,
        use_fp16=config["benchmark"]["use_fp16"],
    )

    metrics = benchmark_model(
        model=model,
        dataloader=dataloader,
        device=device,
        warmup_steps=config["benchmark"]["warmup_steps"],
        measure_steps=config["benchmark"]["measure_steps"],
        use_fp16=config["benchmark"]["use_fp16"],
    )
    metrics.update(
        {
            "experiment_name": config["experiment_name"],
            "device": str(device),
            "use_fp16": bool(config["benchmark"]["use_fp16"]),
            "scale": config["data"]["scale"],
        }
    )

    output_path = Path(config["output"]["benchmark_metrics_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

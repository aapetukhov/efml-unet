from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_dataset
from src.metrics import compute_psnr, compute_ssim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bicubic_upsample(lr_images: torch.Tensor, scale: int) -> torch.Tensor:
    """Upsample images using bicubic interpolation."""
    b, c, h, w = lr_images.shape
    return F.interpolate(lr_images, size=(h * scale, w * scale), mode='bicubic', align_corners=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    dataset = build_dataset(
        lr_dir=config["data"]["val_lr_dir"],
        hr_dir=config["data"]["val_hr_dir"],
        scale=config["data"]["scale"],
        crop_size=config["data"]["eval_crop_size"],
        training=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=False,
    )

    psnr_scores = []
    ssim_scores = []

    print("Evaluating bicubic interpolation...")
    with torch.inference_mode():
        for i, (lr_images, hr_images, _) in enumerate(dataloader):
            # Move to CPU since we're not using GPU for bicubic
            lr_images = lr_images.cpu()
            hr_images = hr_images.cpu()

            predictions = bicubic_upsample(lr_images, config["data"]["scale"]).float().clamp(0.0, 1.0).cpu()
            targets = hr_images.float().cpu()

            psnr_scores.append(compute_psnr(predictions[0], targets[0]))
            ssim_scores.append(compute_ssim(predictions[0], targets[0]))

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")

    metrics = {
        "experiment_name": "bicubic_baseline",
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "num_images": len(dataset),
    }

    output_path = Path("./results/bicubic_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()

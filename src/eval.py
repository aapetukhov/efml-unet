from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
        batch_size=1,
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

    psnr_scores = []
    ssim_scores = []
    with torch.inference_mode():
        for lr_images, hr_images, _ in dataloader:
            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)
            if config["benchmark"]["use_fp16"] and device.type == "cuda":
                lr_images = lr_images.half()

            predictions = model(lr_images).float().cpu()
            targets = hr_images.float().cpu()
            psnr_scores.append(compute_psnr(predictions[0], targets[0]))
            ssim_scores.append(compute_ssim(predictions[0], targets[0]))

    metrics = {
        "experiment_name": config["experiment_name"],
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "num_images": len(dataset),
    }

    output_path = Path(config["output"]["eval_metrics_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

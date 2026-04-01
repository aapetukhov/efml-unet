from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_dataset
from src.metrics import compute_psnr, compute_ssim
from src.modeling import build_model


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


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


@torch.inference_mode()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation pass and return loss / PSNR / SSIM."""
    model.eval()
    total_loss = 0.0
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []

    for lr_images, hr_images, _ in val_loader:
        lr_images = lr_images.to(device, non_blocking=True)
        hr_images = hr_images.to(device, non_blocking=True)

        predictions = model(lr_images)
        total_loss += criterion(predictions, hr_images).item()

        # Clamp for metric computation (model no longer clamps internally)
        predictions = predictions.clamp(0.0, 1.0).float().cpu()
        targets = hr_images.float().cpu()
        for pred, tgt in zip(predictions, targets):
            psnr_scores.append(compute_psnr(pred, tgt))
            ssim_scores.append(compute_ssim(pred, tgt))

    num_batches = max(len(val_loader), 1)
    return {
        "val_loss": total_loss / num_batches,
        "val_psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "val_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["benchmark"]["device"])

    train_dataset = build_dataset(
        lr_dir=config["data"]["train_lr_dir"],
        hr_dir=config["data"]["train_hr_dir"],
        scale=config["data"]["scale"],
        crop_size=config["data"]["train_crop_size"],
        training=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        scale=config["data"]["scale"],
    ).to(device)

    criterion = CharbonnierLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["train"]["min_lr"],
    )

    grad_clip_norm: float | None = config["train"].get("grad_clip_norm", None)

    history: list[Dict[str, Any]] = []
    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for lr_images, hr_images, _ in train_loader:
            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(lr_images)
            loss = criterion(predictions, hr_images)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "lr": current_lr,
        }
        history.append(record)
        print(json.dumps(record))
        scheduler.step()

    save_path = Path(config["train"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        save_path,
    )


if __name__ == "__main__":
    main()

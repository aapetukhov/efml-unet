from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_dataset
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["benchmark"]["device"])

    train_dataset = build_dataset(
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
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    history = []
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
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        history.append({"epoch": epoch + 1, "train_l1": epoch_loss})
        print(json.dumps(history[-1]))

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

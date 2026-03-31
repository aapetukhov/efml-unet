from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader

from src.datasets.collate import sr_collate_fn
from src.datasets.sr_dataset import SuperResolutionDataset
from src.transforms.augment import build_pair_augment
from src.transforms.normalize import build_tensor_transform
from src.transforms.scale import build_pair_transform


def _resolve_dir(path_like) -> Path:
    path = Path(to_absolute_path(str(path_like)))
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return path


def build_dataset(split_cfg, common_cfg, transforms_cfg, training: bool):
    hr_dir = _resolve_dir(split_cfg.hr_dir)
    resize_transform = build_pair_transform(transforms_cfg, training=training)
    augment = build_pair_augment(transforms_cfg, training=training)

    def pair_transform(lr_img, hr_img):
        if resize_transform:
            lr_img, hr_img = resize_transform(lr_img, hr_img)
        if augment:
            lr_img, hr_img = augment(lr_img, hr_img)
        return lr_img, hr_img

    tensor_transform = build_tensor_transform(transforms_cfg)
    return SuperResolutionDataset(
        hr_dir=hr_dir,
        scale=common_cfg.scale,
        crop_size=split_cfg.crop_size,
        random_crop=bool(split_cfg.random_crop),
        pair_transform=pair_transform,
        tensor_transform=tensor_transform,
    )


def get_dataloaders(config, device) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    dataloaders: Dict[str, DataLoader] = {}
    batch_transforms: Dict[str, object] = {}

    common_cfg = config.datasets
    transforms_cfg = config.transforms
    loader_cfg = config.dataloader

    for split_name in ["train", "val", "test"]:
        split_cfg = getattr(common_cfg, split_name, None)
        if split_cfg is None or split_cfg.hr_dir is None:
            continue
        # skip missing folders silently to keep config flexible
        hr_dir = Path(to_absolute_path(str(split_cfg.hr_dir)))
        if not hr_dir.exists():
            continue

        dataset = build_dataset(split_cfg, common_cfg, transforms_cfg, training=split_name == "train")
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=split_name == "train",
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory and device.startswith("cuda"),
            prefetch_factor=getattr(loader_cfg, "prefetch_factor", None),
            persistent_workers=getattr(loader_cfg, "persistent_workers", False),
            collate_fn=sr_collate_fn,
        )

    return dataloaders, batch_transforms

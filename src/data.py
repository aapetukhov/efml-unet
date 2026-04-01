from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class PairedSuperResolutionDataset(Dataset):
    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        scale: int,
        crop_size: Optional[int] = None,
        training: bool = False,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.training = training

        self.lr_paths = sorted(
            path for path in self.lr_dir.rglob("*") if path.suffix.lower() in VALID_EXTENSIONS
        )
        if not self.lr_paths:
            raise RuntimeError(f"No LR images found in {self.lr_dir}")

        self.samples: list[tuple[Path, Path, str]] = []
        for lr_path in self.lr_paths:
            sample_name = self._normalize_lr_name(lr_path.stem)
            hr_path = self.hr_dir / f"{sample_name}.png"
            if not hr_path.exists():
                hr_path = self.hr_dir / f"{sample_name}.jpg"
            if not hr_path.exists():
                raise RuntimeError(f"Missing HR pair for {lr_path.name} in {self.hr_dir}")
            self.samples.append((lr_path, hr_path, sample_name))

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_lr_name(self, stem: str) -> str:
        suffix = f"x{self.scale}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
        return stem

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _paired_random_crop(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if self.crop_size is None:
            return lr_image, hr_image

        lr_crop_size = min(
            self.crop_size,
            lr_image.size[0],
            lr_image.size[1],
            hr_image.size[0] // self.scale,
            hr_image.size[1] // self.scale,
        )
        # Round down to nearest multiple of 4 so that two MaxPool2d(2)
        # layers in the U-Net encoder produce integer spatial dims and
        # the decoder upsample can exactly match the skip connections.
        lr_crop_size = (lr_crop_size // 4) * 4
        if lr_crop_size < 4:
            lr_crop_size = 4
        top, left, height, width = RandomCrop.get_params(
            lr_image,
            output_size=(lr_crop_size, lr_crop_size),
        )
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_height = height * self.scale
        hr_width = width * self.scale
        return (
            TF.crop(lr_image, top, left, height, width),
            TF.crop(hr_image, hr_top, hr_left, hr_height, hr_width),
        )

    def _paired_center_crop(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if self.crop_size is None:
            return lr_image, hr_image

        lr_crop_size = min(
            self.crop_size,
            lr_image.size[0],
            lr_image.size[1],
            hr_image.size[0] // self.scale,
            hr_image.size[1] // self.scale,
        )
        # Round down to nearest multiple of 4 (see _paired_random_crop).
        lr_crop_size = (lr_crop_size // 4) * 4
        if lr_crop_size < 4:
            lr_crop_size = 4
        hr_crop_size = lr_crop_size * self.scale
        return (
            TF.center_crop(lr_image, [lr_crop_size, lr_crop_size]),
            TF.center_crop(hr_image, [hr_crop_size, hr_crop_size]),
        )

    def _augment(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if torch_rand() < 0.5:
            lr_image = TF.hflip(lr_image)
            hr_image = TF.hflip(hr_image)
        if torch_rand() < 0.5:
            lr_image = TF.vflip(lr_image)
            hr_image = TF.vflip(hr_image)

        rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        rotation = rotations[int(torch_rand() * len(rotations))]
        if rotation is not None:
            lr_image = lr_image.transpose(rotation)
            hr_image = hr_image.transpose(rotation)
        return lr_image, hr_image

    def __getitem__(self, index: int):
        lr_path, hr_path, sample_name = self.samples[index]
        lr_image = self._load_image(lr_path)
        hr_image = self._load_image(hr_path)

        if self.training:
            lr_image, hr_image = self._paired_random_crop(lr_image, hr_image)
            lr_image, hr_image = self._augment(lr_image, hr_image)
        else:
            lr_image, hr_image = self._paired_center_crop(lr_image, hr_image)

        lr_tensor = TF.to_tensor(lr_image)
        hr_tensor = TF.to_tensor(hr_image)
        return lr_tensor, hr_tensor, sample_name


def torch_rand() -> float:
    import random

    return random.random()


def build_dataset(
    lr_dir: str | Path,
    hr_dir: str | Path,
    scale: int,
    crop_size: Optional[int],
    training: bool,
):
    return PairedSuperResolutionDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale=scale,
        crop_size=crop_size,
        training=training,
    )

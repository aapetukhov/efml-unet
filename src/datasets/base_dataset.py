from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF

from src.transforms.scale import ensure_divisible

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class BaseSuperResolutionDataset(Dataset):
    """Base dataset that generates LR/HR pairs on the fly.

    The dataset crops HR images to be divisible by `scale`, downsamples them
    to obtain LR images, upsamples LR back with bicubic to form an input close
    to the evaluation protocol, and converts everything to tensors.
    """

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int = 2,
        crop_size: Optional[int] = None,
        random_crop: bool = False,
        interpolation: int = Image.BICUBIC,
        pair_transform: Optional[Callable] = None,
        tensor_transform: Optional[Callable] = None,
    ) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.interpolation = interpolation
        self.pair_transform = pair_transform
        self.tensor_transform = tensor_transform

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {self.hr_dir}")

        self.image_paths = sorted(
            path for path in self.hr_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images with supported extensions found in {self.hr_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_hr(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        image = ensure_divisible(image, self.scale)
        if self.crop_size is None:
            return image
        if self.random_crop:
            top, left, height, width = RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            return TF.crop(image, top, left, height, width)
        # center crop for eval
        return TF.center_crop(image, output_size=[self.crop_size, self.crop_size])

    def _make_lr(self, hr_image: Image.Image) -> Image.Image:
        lr_size = (hr_image.size[0] // self.scale, hr_image.size[1] // self.scale)
        return hr_image.resize(lr_size, self.interpolation)

    def __getitem__(self, index: int) -> Dict[str, object]:
        hr_image = self._load_hr(self.image_paths[index])
        lr_image = self._make_lr(hr_image)
        lr_upscaled = lr_image.resize(hr_image.size, self.interpolation)

        if self.pair_transform is not None:
            lr_upscaled, hr_image = self.pair_transform(lr_upscaled, hr_image)

        lr_tensor = TF.to_tensor(lr_upscaled)
        hr_tensor = TF.to_tensor(hr_image)

        if self.tensor_transform is not None:
            lr_tensor, hr_tensor = self.tensor_transform(lr_tensor, hr_tensor)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "name": self.image_paths[index].name,
            "scale": self.scale,
        }

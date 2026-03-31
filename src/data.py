from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        hr_dir: str | Path,
        scale: int = 2,
        crop_size: Optional[int] = None,
        training: bool = False,
        interpolation=Image.BICUBIC,
    ) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.training = training
        self.interpolation = interpolation
        self.image_paths = sorted(
            path for path in self.hr_dir.iterdir()
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.hr_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_hr(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        width = width - (width % self.scale)
        height = height - (height % self.scale)
        image = image.crop((0, 0, width, height))
        return image

    def _random_crop(self, image: Image.Image) -> Image.Image:
        if self.crop_size is None:
            return image
        crop_size = min(self.crop_size, image.size[0], image.size[1])
        top, left, height, width = RandomCrop.get_params(
            image,
            output_size=(crop_size, crop_size),
        )
        return TF.crop(image, top, left, height, width)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        if self.crop_size is None:
            return image
        crop_size = min(self.crop_size, image.size[0], image.size[1])
        return TF.center_crop(image, [crop_size, crop_size])

    def _make_lr(self, hr_image: Image.Image) -> Image.Image:
        lr_size = (
            hr_image.size[0] // self.scale,
            hr_image.size[1] // self.scale,
        )
        return hr_image.resize(lr_size, self.interpolation)

    def __getitem__(self, index: int):
        hr_image = self._load_hr(self.image_paths[index])
        if self.training:
            hr_image = self._random_crop(hr_image)
        else:
            hr_image = self._center_crop(hr_image)

        lr_image = self._make_lr(hr_image)
        lr_upscaled = lr_image.resize(hr_image.size, self.interpolation)

        lr_tensor = TF.to_tensor(lr_upscaled)
        hr_tensor = TF.to_tensor(hr_image)
        return lr_tensor, hr_tensor, self.image_paths[index].name


def build_dataset(
    hr_dir: str | Path,
    scale: int,
    crop_size: Optional[int],
    training: bool,
):
    return SuperResolutionDataset(
        hr_dir=hr_dir,
        scale=scale,
        crop_size=crop_size,
        training=training,
    )

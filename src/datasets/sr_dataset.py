from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.datasets.base_dataset import BaseSuperResolutionDataset


class SuperResolutionDataset(BaseSuperResolutionDataset):
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
        super().__init__(
            hr_dir=hr_dir,
            scale=scale,
            crop_size=crop_size,
            random_crop=random_crop,
            interpolation=interpolation,
            pair_transform=pair_transform,
            tensor_transform=tensor_transform,
        )

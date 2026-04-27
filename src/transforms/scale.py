from __future__ import annotations

from typing import Callable, Optional, Tuple

from PIL import Image


def ensure_divisible(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    width = width - (width % scale)
    height = height - (height % scale)
    return image.crop((0, 0, width, height))


def build_pair_transform(config, training: bool = False) -> Optional[Callable]:
    resize_cfg = getattr(config, "resize_before_crop", None)
    if resize_cfg is None:
        return None

    def _transform(lr_image: Image.Image, hr_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        target_size = (resize_cfg[1], resize_cfg[0]) if isinstance(resize_cfg, (list, tuple)) else resize_cfg
        return (
            lr_image.resize(target_size, Image.BICUBIC),
            hr_image.resize(target_size, Image.BICUBIC),
        )

    return _transform

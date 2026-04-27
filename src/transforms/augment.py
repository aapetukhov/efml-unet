from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


class PairAugmentor:
    def __init__(self, config):
        self.hflip_p = getattr(config, "hflip_prob", 0.0)
        self.vflip_p = getattr(config, "vflip_prob", 0.0)
        self.r90_p = getattr(config, "rotate90_prob", 0.0)
        self.transpose_p = getattr(config, "transpose_prob", 0.0)
        self.cj_p = getattr(config, "color_jitter_prob", 0.0)
        cj_cfg = getattr(config, "color_jitter", {})
        self.cj_b = cj_cfg.get("brightness", 0.0)
        self.cj_c = cj_cfg.get("contrast", 0.0)
        self.cj_s = cj_cfg.get("saturation", 0.0)
        self.cj_h = cj_cfg.get("hue", 0.0)
        self.gn_p = getattr(config, "gaussian_noise_prob", 0.0)
        self.gn_std = getattr(config, "gaussian_noise_std", 0.0)
        self.cutout_p = getattr(config, "cutout_prob", 0.0)
        self.cutout_min = getattr(config, "cutout_min_ratio", 0.0)
        self.cutout_max = getattr(config, "cutout_max_ratio", 0.0)

    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.hflip_p:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.vflip_p:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < self.r90_p:
            lr_img = lr_img.transpose(Image.ROTATE_90)
            hr_img = hr_img.transpose(Image.ROTATE_90)
        if random.random() < self.transpose_p:
            lr_img = lr_img.transpose(Image.TRANSPOSE)
            hr_img = hr_img.transpose(Image.TRANSPOSE)

        if self.cj_p > 0 and random.random() < self.cj_p:
            lr_img, hr_img = self._color_jitter_pair(lr_img, hr_img)

        if self.gn_p > 0 and self.gn_std > 0 and random.random() < self.gn_p:
            lr_img = self._add_gaussian_noise(lr_img, self.gn_std)
            hr_img = self._add_gaussian_noise(hr_img, self.gn_std)

        if self.cutout_p > 0 and random.random() < self.cutout_p:
            lr_img, hr_img = self._apply_cutout_pair(lr_img, hr_img, self.cutout_min, self.cutout_max)

        return lr_img, hr_img

    def _color_jitter_pair(self, lr_img: Image.Image, hr_img: Image.Image):
        b = 1.0 + random.uniform(-self.cj_b, self.cj_b)
        c = 1.0 + random.uniform(-self.cj_c, self.cj_c)
        s = 1.0 + random.uniform(-self.cj_s, self.cj_s)
        h = random.uniform(-self.cj_h, self.cj_h)

        def apply(img):
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
            return img

        return apply(lr_img), apply(hr_img)

    @staticmethod
    def _add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _apply_cutout_pair(
        lr_img: Image.Image,
        hr_img: Image.Image,
        min_ratio: float,
        max_ratio: float,
    ) -> Tuple[Image.Image, Image.Image]:
        if min_ratio <= 0 or max_ratio <= 0:
            return lr_img, hr_img
        w, h = lr_img.size
        area = w * h
        ratio = random.uniform(min_ratio, max_ratio)
        side = int((area * ratio) ** 0.5)
        side = max(1, min(side, w, h))
        x0 = random.randint(0, w - side)
        y0 = random.randint(0, h - side)
        x1, y1 = x0 + side, y0 + side

        def mask(img: Image.Image):
            img = img.copy()
            draw = ImageDraw.Draw(img)
            # neutral gray fill to avoid heavy color shift
            draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
            return img

        return mask(lr_img), mask(hr_img)


def build_pair_augment(config, training: bool) -> Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]]:
    if not training:
        return None
    aug = PairAugmentor(config)
    # if all probs are zero, skip
    if (
        aug.hflip_p == 0
        and aug.vflip_p == 0
        and aug.r90_p == 0
        and aug.cj_p == 0
        and aug.gn_p == 0
        and aug.transpose_p == 0
        and aug.cutout_p == 0
    ):
        return None
    return aug

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from torchvision.utils import save_image


def save_batch_images(tensor_batch: torch.Tensor, names: Iterable[str], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for tensor, name in zip(tensor_batch, names):
        path = output_dir / name
        save_image(tensor.clamp(0.0, 1.0), path)
        saved_paths.append(path)
    return saved_paths

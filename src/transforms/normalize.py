from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
from torchvision.transforms import functional as TF


def _normalize_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    lr = TF.normalize(lr, mean=mean, std=std)
    hr = TF.normalize(hr, mean=mean, std=std)
    return lr, hr


def build_tensor_transform(config) -> Optional[Callable]:
    if not getattr(config, "normalize", False):
        return None
    mean = getattr(config, "mean", [0.5, 0.5, 0.5])
    std = getattr(config, "std", [0.5, 0.5, 0.5])
    return lambda lr, hr: _normalize_pair(lr, hr, mean=mean, std=std)

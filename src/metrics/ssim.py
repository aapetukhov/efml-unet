from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.metrics.base_metric import BaseMetric


class SSIMMetric(BaseMetric):
    name = "ssim"

    def __init__(self, channel_axis: int = 2) -> None:
        self.channel_axis = channel_axis

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction_np = prediction.detach().cpu().numpy().transpose(1, 2, 0)
        target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
        prediction_np = np.clip(prediction_np, 0.0, 1.0)
        target_np = np.clip(target_np, 0.0, 1.0)
        return float(
            structural_similarity(
                prediction_np,
                target_np,
                channel_axis=self.channel_axis,
                data_range=1.0,
            )
        )

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    """L1 pixel loss + VGG-16 feature loss for sharper SR outputs.

    VGG weights are frozen — only prediction/target pass through them.
    Inputs are expected in [0, 1]; internally normalized to ImageNet stats
    before VGG forward.

    Args:
        l1_weight: pixel-level L1 weight
        perceptual_weight: VGG feature L1 weight (scale: ~0.01-0.1)
        feature_layer: 'relu2_2' (finer texture) or 'relu3_3' (richer semantics)
    """

    _LAYER_IDX = {"relu2_2": 9, "relu3_3": 16}

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        feature_layer: str = "relu3_3",
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        cutoff = self._LAYER_IDX.get(feature_layer, 16)
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:cutoff])
        for p in self.features.parameters():
            p.requires_grad = False

        self.l1 = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _vgg_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        l1_val = self.l1(prediction, target)

        pred_feats = self.features(self._vgg_normalize(prediction.clamp(0.0, 1.0)))
        targ_feats = self.features(self._vgg_normalize(target.clamp(0.0, 1.0)))
        perc_val = self.l1(pred_feats, targ_feats)

        total = self.l1_weight * l1_val + self.perceptual_weight * perc_val
        return {"loss": total, "l1": l1_val.detach(), "perceptual": perc_val.detach()}

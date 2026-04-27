from __future__ import annotations

import torch.nn as nn


class L1LossWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target)
        return {"loss": loss}

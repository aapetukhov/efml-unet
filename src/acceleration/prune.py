from __future__ import annotations

import torch.nn as nn
from torch.nn.utils import prune


def apply_global_magnitude_pruning(model: nn.Module, prune_ratio: float) -> None:
    """Zero out the globally lowest-L1 fraction of weights in all 1x1 Conv2d layers.

    Uses torch.nn.utils.prune reparametrization — call make_pruning_permanent
    afterwards to bake masks into weights and remove hooks.
    """
    params = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)
    ]
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio,
    )


def make_pruning_permanent(model: nn.Module) -> None:
    """Remove reparametrization hooks and bake masks into weights."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass

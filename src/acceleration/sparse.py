from __future__ import annotations

import torch
import torch.nn as nn


def _prune_2_4(w: torch.Tensor) -> torch.Tensor:
    """Zero out 2 smallest-magnitude values in every group of 4 (last dim)."""
    t = w.view(-1, 4)
    _, idx = torch.topk(t.abs(), k=2, dim=-1, largest=False)
    mask = torch.ones_like(t).scatter_(1, idx, 0.0)
    return (t * mask).view_as(w)


def apply_sparse_2_4(
    model: nn.Module,
    convert_to_sparse: bool = False,
) -> list[str]:
    """Apply 2:4 magnitude pruning to all 1x1 Conv2d weights.

    convert_to_sparse=True converts to SparseSemiStructuredTensor for
    cuSPARSELt acceleration (requires Ampere GPU + PyTorch >= 2.1).

    Returns list of pruned layer names.
    """
    pruned: list[str] = []
    for name, m in model.named_modules():
        if not (isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)):
            continue
        w = m.weight.data.view(m.weight.size(0), -1)  # (out, in)
        if w.size(1) % 4 != 0:
            continue
        w_pruned = _prune_2_4(w)
        if convert_to_sparse:
            try:
                from torch.sparse import to_sparse_semi_structured
                m.weight = nn.Parameter(
                    to_sparse_semi_structured(w_pruned).view_as(m.weight)
                )
            except Exception:
                m.weight.data.copy_(w_pruned.view_as(m.weight))
        else:
            m.weight.data.copy_(w_pruned.view_as(m.weight))
        pruned.append(name)
    return pruned

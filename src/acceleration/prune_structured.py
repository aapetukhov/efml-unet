from __future__ import annotations

import torch
import torch.nn as nn

from src.model.heavy_model import MBResBlock


def _prune_mbresblock(block: MBResBlock, prune_ratio: float, num_groups: int = 8) -> int:
    # Only mid_ch shrinks — in_ch, out_ch, skip untouched → U-Net connections stay valid.
    expand_w = block.expand[2].weight.data  # (mid_ch, in_ch, 1, 1)
    mid_ch = expand_w.size(0)
    in_ch = expand_w.size(1)
    out_ch = block.project[0].weight.size(0)

    importance = expand_w.abs().sum(dim=[1, 2, 3])  # (mid_ch,)

    n_keep = int(mid_ch * (1.0 - prune_ratio))
    n_keep = max((n_keep // num_groups) * num_groups, num_groups)

    _, keep_idx = torch.topk(importance, k=n_keep)
    keep_idx = keep_idx.sort().values

    new_exp_conv = nn.Conv2d(in_ch, n_keep, 1, bias=False)
    new_exp_conv.weight.data = expand_w[keep_idx]
    block.expand[2] = new_exp_conv

    old_dw_gn: nn.GroupNorm = block.dw_conv[0]
    new_dw_gn = nn.GroupNorm(num_groups, n_keep)
    new_dw_gn.weight.data = old_dw_gn.weight.data[keep_idx]
    new_dw_gn.bias.data = old_dw_gn.bias.data[keep_idx]
    block.dw_conv[0] = new_dw_gn

    old_dw_conv: nn.Conv2d = block.dw_conv[2]
    new_dw_conv = nn.Conv2d(n_keep, n_keep, 3, padding=1, groups=n_keep, bias=False)
    new_dw_conv.weight.data = old_dw_conv.weight.data[keep_idx]
    block.dw_conv[2] = new_dw_conv

    old_proj_conv: nn.Conv2d = block.project[0]
    new_proj_conv = nn.Conv2d(n_keep, out_ch, 1, bias=False)
    new_proj_conv.weight.data = old_proj_conv.weight.data[:, keep_idx, :, :]
    block.project[0] = new_proj_conv

    return n_keep


def prune_all_mbresblocks(
    model: nn.Module,
    prune_ratio: float,
    num_groups: int = 8,
) -> dict[str, int]:
    """Prune expand channels in every MBResBlock of the model in-place.

    Returns {block_name: new_mid_ch} for logging.
    """
    results: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, MBResBlock):
            new_mid = _prune_mbresblock(module, prune_ratio, num_groups)
            results[name] = new_mid
    return results

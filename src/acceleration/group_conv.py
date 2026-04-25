from __future__ import annotations

import torch
import torch.nn as nn

from src.model.heavy_model import MBResBlock


def _to_grouped(old_conv: nn.Conv2d, G: int) -> nn.Conv2d:
    """Replace a 1x1 Conv2d with a grouped version, keeping block-diagonal weights.

    The original weight W of shape (out_ch, in_ch, 1, 1) is treated as a matrix.
    Group g handles output channels [g*out_pg : (g+1)*out_pg] and input channels
    [g*in_pg : (g+1)*in_pg].  We initialise with the corresponding block-diagonal
    sub-matrix so finetuning starts close to the original function (for the
    intra-group interactions that already existed).

    Requires out_ch % G == 0 and in_ch % G == 0.
    """
    out_ch, in_ch = old_conv.weight.size(0), old_conv.weight.size(1)
    assert out_ch % G == 0 and in_ch % G == 0
    out_pg, in_pg = out_ch // G, in_ch // G

    device = old_conv.weight.device
    new_conv = nn.Conv2d(in_ch, out_ch, 1, groups=G, bias=False).to(device)
    w_old = old_conv.weight.data          # (out_ch, in_ch, 1, 1)
    w_new = new_conv.weight.data.zero_()  # (out_ch, in_pg,  1, 1)
    for g in range(G):
        w_new[g * out_pg:(g + 1) * out_pg] = \
            w_old[g * out_pg:(g + 1) * out_pg, g * in_pg:(g + 1) * in_pg]
    return new_conv


def convert_to_grouped_conv(
    model: nn.Module,
    groups: int = 2,
    target: str = "expand",     # "expand" | "project" | "both"
) -> dict[str, str]:
    """Replace 1x1 Conv2d in every MBResBlock with grouped convolutions in-place.

    target:
        "expand"  — only expand pw conv (C_in → mid_ch)
        "project" — only project pw conv (mid_ch → C_out)
        "both"    — both expand and project

    Returns {block_name: description} for logging.
    """
    log: dict[str, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, MBResBlock):
            continue
        parts = []
        if target in ("expand", "both"):
            module.expand[2] = _to_grouped(module.expand[2], groups)
            parts.append("expand")
        if target in ("project", "both"):
            module.project[0] = _to_grouped(module.project[0], groups)
            parts.append("project")
        log[name] = f"grouped {'+'.join(parts)} G={groups}"
    return log

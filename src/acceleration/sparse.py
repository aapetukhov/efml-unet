from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heavy_model import MBResBlock


class Pointwise2d(nn.Module):
    """1x1 Conv2d drop-in that routes through nn.Linear.

    SparseSemiStructuredTensor accelerates F.linear via cuSPARSELt, but
    NOT F.conv2d — even with kernel_size=1. Replacing expand/project convs
    with this wrapper is the correct path to hardware 2:4 speedup on Ampere+.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()              # (B, H, W, C_in)
        x = F.linear(x.to(self.linear.weight.dtype), self.linear.weight)  # (B, H, W, C_out)
        return x.permute(0, 3, 1, 2).contiguous().to(orig_dtype)  # (B, C_out, H, W)


def _prune_2_4(w: torch.Tensor) -> torch.Tensor:
    """Zero out 2 smallest-magnitude values in every group of 4 (last dim)."""
    t = w.view(-1, 4)
    _, idx = torch.topk(t.abs(), k=2, dim=-1, largest=False)
    mask = torch.ones_like(t).scatter_(1, idx, 0.0)
    return (t * mask).view_as(w)


def _to_sparse_pointwise(conv: nn.Conv2d) -> Pointwise2d:
    """Prune a 1x1 Conv2d to 2:4 and return Pointwise2d with sparse Linear weight.

    cuSPARSELt requires FP16/BF16; weight is cast to FP16 before conversion.
    Pointwise2d.forward auto-casts the input to match and casts output back,
    so the rest of the model can stay in FP32.  For best speedup, cast the
    whole model to FP16 with model.half() before applying sparse.
    """
    out_ch, in_ch = conv.weight.size(0), conv.weight.size(1)
    w_pruned = _prune_2_4(conv.weight.data.view(out_ch, in_ch)).half()  # FP16 for cuSPARSELt

    from torch.sparse import to_sparse_semi_structured
    linear = nn.Linear(in_ch, out_ch, bias=False)
    linear.weight = nn.Parameter(to_sparse_semi_structured(w_pruned))
    return Pointwise2d(linear).to(conv.weight.device)


def _prune_inplace(conv: nn.Conv2d) -> None:
    w = conv.weight.data.view(conv.weight.size(0), -1)
    if w.size(1) % 4 == 0:
        conv.weight.data.copy_(_prune_2_4(w).view_as(conv.weight))


def apply_sparse_2_4(
    model: nn.Module,
    convert_to_sparse: bool = False,
) -> list[str]:
    """Apply 2:4 magnitude pruning to all 1x1 Conv2d weights.

    convert_to_sparse=True (Ampere+, PyTorch >= 2.1):
      Replaces expand/project convs in every MBResBlock with Pointwise2d
      backed by a SparseSemiStructuredTensor weight — this is the only path
      that triggers cuSPARSELt and yields actual latency speedup.

    convert_to_sparse=False:
      All 1x1 convs are pruned in-place (stays dense, no speedup).
    """
    pruned: list[str] = []

    if convert_to_sparse:
        for name, module in model.named_modules():
            if not isinstance(module, MBResBlock):
                continue
            for seq, idx, label in [
                (module.expand,  2, f"{name}.expand.2"),
                (module.project, 0, f"{name}.project.0"),
            ]:
                try:
                    seq[idx] = _to_sparse_pointwise(seq[idx])
                except Exception:
                    _prune_inplace(seq[idx])  # GPU doesn't support sparse → dense fallback
                pruned.append(label)

    # In-place masking for all remaining 1x1 Conv2d (head, skip connections)
    # When convert_to_sparse=False this covers everything.
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            _prune_inplace(m)
            pruned.append(name)

    return list(dict.fromkeys(pruned))  # deduplicate, preserve order

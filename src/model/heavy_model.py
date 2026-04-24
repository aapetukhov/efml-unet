from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excite channel attention (Hu et al., 2018).

    Acceleration target: fc1 (C → C//ratio) and fc2 (C//ratio → C) weight
    matrices are pruned together by reducing `bottleneck_ratio` — e.g. a
    structured pruner can remove entire columns/rows, or their weights can
    be 2:4-sparsified.
    """

    def __init__(self, channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        reduced = max(channels // bottleneck_ratio, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.squeeze(x).flatten(1)
        return x * self.excite(s).view(x.size(0), -1, 1, 1)


class MBResBlock(nn.Module):
    """Inverted-residual block: pw-expand → dw-3x3 → pw-project + residual.

    Acceleration targets by component:
    - expand 1x1  (C → C*expand_ratio): large dense matrix → 2:4 sparsity,
                                         low-rank factorisation, group convolution
    - dw 3x3      (C*r, groups=C*r):    already factorised spatially; skip for
                                         2:4 (too few params per channel)
    - project 1x1 (C*r → C_out):        same as expand — 2:4, factorisation

    GroupNorm: no global batch statistics → safe to prune/remove channels without
    re-running batch statistics.  Constraint: in_ch and out_ch must be divisible
    by `num_groups` (guaranteed by SRUNetHeavy's stem projection).
    """

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4, num_groups: int = 8):
        super().__init__()
        mid_ch = in_ch * expand_ratio

        self.expand = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
        )
        self.dw_conv = nn.Sequential(
            nn.GroupNorm(num_groups, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw_conv(out)
        out = self.project(out)
        return out + self.skip(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, num_groups):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = MBResBlock(in_ch, out_ch, expand_ratio, num_groups)

    def forward(self, x):
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, expand_ratio, num_groups):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = MBResBlock(in_ch + skip_ch, out_ch, expand_ratio, num_groups)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.block(torch.cat([skip, x], dim=1))


class SRUNetHeavy(nn.Module):
    """Heavy U-Net baseline built for inference-acceleration experiments.

    Design intent
    -------------
    Heavier than SRUNet (more params, more FLOPs) so that post-training
    acceleration methods have room to show meaningful speedup:

    - 2:4 semi-structured sparsity  (Andrey): target expand/project 1x1 Conv
    - Structured channel pruning    (Artem):  use SE attention magnitudes as scores
    - expand_ratio reduction:                 fine-tune with lower expand_ratio
    - Group convolution replacement:          split expand pointwise into G groups

    Architecture
    ------------
    stem → enc x 4 → bottleneck (MBRes + SE) → dec x 4 → head + global skip

    All channels are multiples of base_channels, so num_groups always divides
    them exactly.  No BatchNorm → channel removal never breaks running statistics.

    Parameters (defaults: base_channels=96, expand_ratio=4)
    ---------------------------------------------------------
    ~31 M parameters vs 17 M for SRUNet (base_channels=64, DoubleConv)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 96,
        expand_ratio: int = 4,
        se_bottleneck_ratio: int = 4,
        residual: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()
        assert base_channels % num_groups == 0, (
            f"base_channels ({base_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.residual = residual and in_channels == out_channels
        bc = base_channels
        r, g = expand_ratio, num_groups

        # Stem: plain conv to project input channels → base_channels
        # (keeps GroupNorm valid for all subsequent MBResBlocks)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, padding=1, bias=False),
            nn.GroupNorm(g, bc),
            nn.GELU(),
        )

        # Encoder
        self.inc   = MBResBlock(bc,    bc,    r, g)
        self.down1 = _Down(bc,    bc*2,  r, g)
        self.down2 = _Down(bc*2,  bc*4,  r, g)
        self.down3 = _Down(bc*4,  bc*8,  r, g)
        self.down4 = _Down(bc*8,  bc*8,  r, g)   # keep channels (bilinear mode)

        # Bottleneck: extra MBResBlock + SE attention
        self.bottleneck = nn.Sequential(
            MBResBlock(bc*8, bc*8, r, g),
            SEBlock(bc*8, se_bottleneck_ratio),
        )

        # Decoder (concat: upsampled + skip → block)
        self.up1 = _Up(bc*8,  bc*8,  bc*4, r, g)   # bc*16 in → bc*4 out
        self.up2 = _Up(bc*4,  bc*4,  bc*2, r, g)   # bc*8  in → bc*2 out
        self.up3 = _Up(bc*2,  bc*2,  bc,   r, g)   # bc*4  in → bc   out
        self.up4 = _Up(bc,    bc,    bc,   r, g)   # bc*2  in → bc   out

        self.head = nn.Conv2d(bc, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        x1 = self.inc(s)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.head(out)
        if self.residual:
            out = out + x
        return out

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Pre-activation style residual block: act → conv → act → conv + skip.

    The identity path is kept clean (no activation after the sum) so that
    gradients flow freely through the skip connection.  ``conv2`` weights are
    initialised to zero so that each block starts as an identity mapping.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, a=0.2, nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers.extend(ResBlock(out_channels) for _ in range(num_blocks))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers: list[nn.Module] = [
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers.extend(ResBlock(out_channels) for _ in range(num_blocks))
        self.fuse = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class UNetSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1 or scale & (scale - 1) != 0:
            raise ValueError("scale must be a positive power of 2")

        self.scale = scale
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.enc1 = EncoderStage(base_channels, base_channels, num_blocks=2)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = EncoderStage(base_channels, base_channels * 2, num_blocks=2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = EncoderStage(base_channels * 2, base_channels * 4, num_blocks=4)

        self.dec2 = DecoderStage(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            num_blocks=2,
        )
        self.dec1 = DecoderStage(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            num_blocks=2,
        )

        self.lr_refinement = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )

        upsampling_layers: list[nn.Module] = []
        current_scale = scale
        while current_scale > 1:
            upsampling_layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResBlock(base_channels),
                ]
            )
            current_scale //= 2
        self.upsampling_head = nn.Sequential(*upsampling_layers)

        self.reconstruction = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input_proj(x)

        enc1 = self.enc1(x0)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.dec2(bottleneck, enc2)
        dec1 = self.dec1(dec2, enc1)

        refined_lr = self.lr_refinement(dec1)
        upsampled_features = self.upsampling_head(refined_lr)
        bicubic = torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        prediction = self.reconstruction(upsampled_features)
        return prediction + bicubic


def build_model(
    in_channels: int,
    out_channels: int,
    base_channels: int,
    scale: int,
) -> nn.Module:
    return UNetSR(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        scale=scale,
    )


def prepare_model_for_inference(
    model: nn.Module,
    device: torch.device,
    use_fp16: bool,
) -> nn.Module:
    model = model.to(device)
    model.eval()
    if use_fp16 and device.type == "cuda":
        model = model.half()
    return model

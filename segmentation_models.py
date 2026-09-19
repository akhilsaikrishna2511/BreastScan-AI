"""Attention U-Net++ segmentation model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolutional layers with batch normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate that suppresses irrelevant skip-connection activations."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != gate.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * attention


class NestedDecoderBlock(nn.Module):
    """One U-Net++ nested decoder node with attention-gated skip inputs."""

    def __init__(self, decoder_channels: int, skip_channels: list[int], out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gates = nn.ModuleList(
            [
                AttentionGate(
                    skip_channels=channels,
                    gate_channels=out_channels,
                    inter_channels=max(out_channels // 2, 8),
                )
                for channels in skip_channels
            ]
        )
        self.conv = ConvBlock(out_channels + sum(skip_channels), out_channels)

    def forward(self, decoder: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        upsampled = self.up(decoder)
        gated_skips: list[torch.Tensor] = []
        for skip, gate in zip(skips, self.gates):
            if skip.shape[2:] != upsampled.shape[2:]:
                skip = F.interpolate(skip, size=upsampled.shape[2:], mode="bilinear", align_corners=False)
            gated_skips.append(gate(skip, upsampled))
        return self.conv(torch.cat([*gated_skips, upsampled], dim=1))


class AttentionUNetPlusPlus(nn.Module):
    """Attention U-Net++ for binary tumor segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        filters = [base_channels * (2**i) for i in range(5)]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels, filters[0], dropout)
        self.conv1_0 = ConvBlock(filters[0], filters[1], dropout)
        self.conv2_0 = ConvBlock(filters[1], filters[2], dropout)
        self.conv3_0 = ConvBlock(filters[2], filters[3], dropout)
        self.conv4_0 = ConvBlock(filters[3], filters[4], dropout)

        self.conv0_1 = NestedDecoderBlock(filters[1], [filters[0]], filters[0])
        self.conv1_1 = NestedDecoderBlock(filters[2], [filters[1]], filters[1])
        self.conv2_1 = NestedDecoderBlock(filters[3], [filters[2]], filters[2])
        self.conv3_1 = NestedDecoderBlock(filters[4], [filters[3]], filters[3])

        self.conv0_2 = NestedDecoderBlock(filters[1], [filters[0], filters[0]], filters[0])
        self.conv1_2 = NestedDecoderBlock(filters[2], [filters[1], filters[1]], filters[1])
        self.conv2_2 = NestedDecoderBlock(filters[3], [filters[2], filters[2]], filters[2])

        self.conv0_3 = NestedDecoderBlock(filters[1], [filters[0], filters[0], filters[0]], filters[0])
        self.conv1_3 = NestedDecoderBlock(filters[2], [filters[1], filters[1], filters[1]], filters[1])

        self.conv0_4 = NestedDecoderBlock(filters[1], [filters[0], filters[0], filters[0], filters[0]], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(x1_0, [x0_0])
        x1_1 = self.conv1_1(x2_0, [x1_0])
        x2_1 = self.conv2_1(x3_0, [x2_0])
        x3_1 = self.conv3_1(x4_0, [x3_0])

        x0_2 = self.conv0_2(x1_1, [x0_0, x0_1])
        x1_2 = self.conv1_2(x2_1, [x1_0, x1_1])
        x2_2 = self.conv2_2(x3_1, [x2_0, x2_1])

        x0_3 = self.conv0_3(x1_2, [x0_0, x0_1, x0_2])
        x1_3 = self.conv1_3(x2_2, [x1_0, x1_1, x1_2])

        x0_4 = self.conv0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])

        if self.deep_supervision:
            return torch.stack(
                [
                    self.final1(x0_1),
                    self.final2(x0_2),
                    self.final3(x0_3),
                    self.final4(x0_4),
                ],
                dim=0,
            ).mean(dim=0)
        return self.final4(x0_4)


def build_segmentation_model(config: dict) -> AttentionUNetPlusPlus:
    """Build an Attention U-Net++ from a config dictionary."""

    return AttentionUNetPlusPlus(
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        base_channels=config.get("base_channels", 32),
        dropout=config.get("dropout", 0.0),
        deep_supervision=config.get("deep_supervision", False),
    )

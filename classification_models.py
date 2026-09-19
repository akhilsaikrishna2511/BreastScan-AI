"""EfficientNet-B3 classifiers with CBAM attention and optional feature fusion."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class ChannelAttention(nn.Module):
    """Channel attention from CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        max_values = torch.amax(x, dim=(2, 3), keepdim=True)
        return x * self.sigmoid(self.mlp(avg) + self.mlp(max_values))


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values = torch.amax(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class EfficientNetB3CBAM(nn.Module):
    """EfficientNet-B3 image classifier with CBAM before global pooling."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.embedding_dim = backbone.classifier[1].in_features
        self.cbam = CBAM(self.embedding_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        return torch.flatten(self.pool(x), 1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        if return_features:
            return logits, embedding
        return logits


class EfficientNetB3CBAMFusion(nn.Module):
    """Image classifier that fuses CBAM EfficientNet embeddings with radiomics features."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.image_branch = EfficientNetB3CBAM(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
        embedding_dim = self.image_branch.embedding_dim
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + fusion_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_branch.forward_features(image)
        tabular = self.feature_branch(tabular_features)
        return self.classifier(torch.cat([image_features, tabular], dim=1))


def build_classifier(config: dict, feature_dim: int | None = None) -> nn.Module:
    """Build an image or fusion classifier from config."""

    if feature_dim is not None:
        return EfficientNetB3CBAMFusion(
            feature_dim=feature_dim,
            num_classes=config.get("num_classes", 3),
            pretrained=config.get("pretrained", True),
            dropout=config.get("dropout", 0.3),
            fusion_hidden=config.get("fusion_hidden", 256),
        )
    return EfficientNetB3CBAM(
        num_classes=config.get("num_classes", 3),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
    )

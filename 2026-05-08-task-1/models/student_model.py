# -*- coding: utf-8 -*-
"""FER System - Improved MobileNetV3-Small Student Model

Student model based on MobileNetV3-Small with SE, CBAM, and ASPP
attention modules for facial expression recognition.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from models.attention import CBAM, SEModule
from models.aspp import ASPP
from models.student_model_config import (
    ASPP_DILATIONS,
    ASPP_OUT_CHANNELS,
    CBAM_CHANNEL_REDUCTION,
    CBAM_SPATIAL_KERNEL,
    CLASSIFIER_HIDDEN_DIM,
    DROPOUT_FINAL,
    DROPOUT_FIRST,
    NUM_CLASSES,
    SE_REDUCTION_RATIO,
)


class LightweightClassifier(nn.Module):
    """Lightweight classification head.

    Uses global average pooling followed by a small FC network
    to keep parameter count low.

    Args:
        in_channels: Number of input feature channels.
        hidden_dim: Hidden dimension of the FC layer.
        num_classes: Number of output classes.
        dropout_first: Dropout rate after pooling.
        dropout_final: Dropout rate before the final linear layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout_first: float = DROPOUT_FIRST,
        dropout_final: float = DROPOUT_FINAL,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_first),
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_final),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Logits [B, num_classes].
        """
        return self.head(x)


class ImprovedMobileNetV3Small(nn.Module):
    """Improved MobileNetV3-Small with attention modules.

    Integrates SE attention on early layers, CBAM on middle/later layers,
    and ASPP for multi-scale context before the classification head.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to load ImageNet pretrained backbone weights.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_FIRST,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load MobileNetV3-Small backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        # Insert SE modules on early layers (0-3)
        self._insert_se_modules()

        # Insert CBAM modules on middle/later layers (4-11)
        self._insert_cbam_modules()

        # ASPP module after feature extraction
        # Last feature layer outputs 576 channels; ASPP takes 96 channels
        # so we need to stop at layer 11 (96 channels)
        self.aspp = ASPP(
            in_channels=ASPP_OUT_CHANNELS,
            out_channels=ASPP_OUT_CHANNELS,
            dilations=ASPP_DILATIONS,
        )

        # Lightweight classifier
        self.classifier = LightweightClassifier(
            in_channels=ASPP_OUT_CHANNELS,
            hidden_dim=CLASSIFIER_HIDDEN_DIM,
            num_classes=num_classes,
            dropout_first=dropout,
            dropout_final=DROPOUT_FINAL,
        )

    def _insert_se_modules(self) -> None:
        """Insert SE attention modules after early feature layers (0-3)."""
        self.se_modules = nn.ModuleList()
        for i in range(4):
            out_channels = self._get_out_channels(i)
            self.se_modules.append(
                SEModule(out_channels, reduction=SE_REDUCTION_RATIO)
            )
        self._se_indices = list(range(4))

    def _insert_cbam_modules(self) -> None:
        """Insert CBAM attention modules on middle/later layers (4-11)."""
        self.cbam_modules = nn.ModuleList()
        for i in range(4, 12):
            out_channels = self._get_out_channels(i)
            self.cbam_modules.append(
                CBAM(out_channels, reduction=CBAM_CHANNEL_REDUCTION,
                     spatial_kernel=CBAM_SPATIAL_KERNEL)
            )
        self._cbam_indices = list(range(4, 12))

    def _get_out_channels(self, layer_idx: int) -> int:
        """Get output channel count for a feature layer.

        Uses the backbone's out_channels attribute when available,
        falling back to a hardcoded channel list.

        Args:
            layer_idx: Index into self.features.

        Returns:
            Number of output channels.
        """
        layer = self.features[layer_idx]
        if hasattr(layer, 'out_channels'):
            return layer.out_channels
        # Fallback channel list for MobileNetV3-Small
        channel_list = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576]
        if layer_idx < len(channel_list):
            return channel_list[layer_idx]
        return 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image [B, 3, 48, 48].

        Returns:
            Logits [B, num_classes].
        """
        # Feature extraction with attention
        # Only use features[0..11] — skip features[12] (Conv2dNormActivation, 576ch)
        se_idx = 0
        cbam_idx = 0

        for i, layer in enumerate(self.features):
            if i >= 12:
                break  # Stop at layer 11 (96ch output)
            x = layer(x)

            if i in self._se_indices:
                x = self.se_modules[se_idx](x)
                se_idx += 1
            elif i in self._cbam_indices:
                x = self.cbam_modules[cbam_idx](x)
                cbam_idx += 1

        # ASPP multi-scale feature extraction
        x = self.aspp(x)

        # Classification
        x = self.classifier(x)
        return x

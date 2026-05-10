# -*- coding: utf-8 -*-
"""FER System - Improved MobileNetV3-Small Student Model

MobileNetV3-Small + SE + CBAM + ASPP for knowledge distillation.
Designed to be trained with a ConvNeXt-Base teacher (70.27% val_acc).
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
    DROPOUT_FIRST,
    DROPOUT_FINAL,
    NUM_CLASSES,
    SE_REDUCTION_RATIO,
)


class LightweightClassifier(nn.Module):
    """Lightweight classification head.

    Global average pooling → dropout → FC → Hardswish → dropout → FC.

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
    """Improved MobileNetV3-Small with SE, CBAM, and ASPP.

    Architecture:
        features[0..11] (stop before layer12/576ch)
        → SE on layers 0-3 → CBAM on layers 4-11
        → ASPP (96ch input, 96ch output)
        → LightweightClassifier (96 → 256 → 7)

    Args:
        num_classes: Number of output classes (FER2013 = 7).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # indices 0-12, last is 576ch

        # ── Attention: SE on early layers (0-3), CBAM on middle layers (4-11) ──
        self.se_modules = nn.ModuleList()
        for i in range(4):
            ch = self._get_out_channels(i)
            self.se_modules.append(SEModule(ch, reduction=SE_REDUCTION_RATIO))

        self.cbam_modules = nn.ModuleList()
        for i in range(4, 12):
            ch = self._get_out_channels(i)
            self.cbam_modules.append(
                CBAM(ch, reduction=CBAM_CHANNEL_REDUCTION,
                     spatial_kernel=CBAM_SPATIAL_KERNEL)
            )

        self._se_indices = list(range(4))
        self._cbam_indices = list(range(4, 12))

        # ── ASPP on 96-channel feature map (after layer11) ──────────
        self.aspp = ASPP(
            in_channels=ASPP_OUT_CHANNELS,
            out_channels=ASPP_OUT_CHANNELS,
            dilations=ASPP_DILATIONS,
        )

        # ── Classifier ───────────────────────────────────────────────
        self.classifier = LightweightClassifier(
            in_channels=ASPP_OUT_CHANNELS,
            hidden_dim=CLASSIFIER_HIDDEN_DIM,
            num_classes=num_classes,
            dropout_first=DROPOUT_FIRST,
            dropout_final=DROPOUT_FINAL,
        )

    def _get_out_channels(self, layer_idx: int) -> int:
        """Get output channel count for a feature layer.

        MobileNetV3-Small features[0..11] channel list:
            [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
        features[12] outputs 576ch (not used in forward).

        Args:
            layer_idx: Index into self.features (0-11).

        Returns:
            Number of output channels.
        """
        layer = self.features[layer_idx]
        if hasattr(layer, "out_channels"):
            return layer.out_channels
        # Fallback – matches features[0..11] exactly
        channel_list = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
        if layer_idx < len(channel_list):
            return channel_list[layer_idx]
        return 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image [B, 3, H, W], recommend 112×112.

        Returns:
            Logits [B, num_classes].
        """
        se_idx = 0
        cbam_idx = 0

        # Feature extraction with interleaved attention (stop before layer12)
        for i, layer in enumerate(self.features):
            if i >= 12:
                break  # stop at layer11 (96ch output)
            x = layer(x)

            if i in self._se_indices:
                x = self.se_modules[se_idx](x)
                se_idx += 1
            elif i in self._cbam_indices:
                x = self.cbam_modules[cbam_idx](x)
                cbam_idx += 1

        # ASPP multi-scale context
        x = self.aspp(x)  # [B, 96, H/32, W/32]

        # Classification
        x = self.classifier(x)
        return x

    @property
    def num_params(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ImprovedMobileNetV3Small(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 112, 112)

    with torch.no_grad():
        logits = model(x)

    print(f"模型参数: {model.num_params:,}")
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {logits.shape}")  # [2, 7]
    print(f"输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

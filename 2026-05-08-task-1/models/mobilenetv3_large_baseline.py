# -*- coding: utf-8 -*-
"""
FER System - MobileNetV3-Large 基线模型

纯净基线，不含任何注意力模块，参数量 ~5.5M。
用于与 MobileNetV3-Small 基线做容量对比。
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from models.student_model_config import NUM_CLASSES


class MobileNetV3LargeBaseline(nn.Module):
    """MobileNetV3-Large 基线模型。

    纯基线结构：标准 MobileNetV3-Large backbone + 轻量分类头。
    参数量约为 Small 的 5 倍，特征提取能力更强。

    Architecture:
        - Backbone: torchvision.models.mobilenet_v3_large (ImageNet 预训练)
        - Features: [B, 3, H, W] → [B, 960, H/32, W/32]
        - Classifier: GAP → Dropout → Linear → Hardswish → Dropout → Linear

    Args:
        num_classes: 输出类别数（FER2013 = 7）。
        pretrained: 是否加载 ImageNet 预训练权重。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        self.features = backbone.features      # [B, 960, H/32, W/32]
        self.avgpool = backbone.avgpool       # GAP: [B, 960, 1, 1]

        # ── 分类头 ──────────────────────────────────────────────────
        # Large backbone 输出 960 通道，比 Small 的 576 宽 ~1.7 倍
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(960, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像 [B, 3, H, W]，支持 48/64/96/112 等任意尺寸。

        Returns:
            logits: [B, num_classes]，未经 softmax 的原始分数。
        """
        x = self.features(x)   # [B, 960, H/32, W/32]
        x = self.avgpool(x)     # [B, 960, 1, 1]
        x = self.classifier(x)  # [B, num_classes]
        return x

    @property
    def num_params(self) -> int:
        """返回模型可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 快速测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MobileNetV3LargeBaseline(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 112, 112)

    with torch.no_grad():
        logits = model(x)

    print(f"模型参数: {model.num_params:,}")
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {logits.shape}")
    print(f"输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

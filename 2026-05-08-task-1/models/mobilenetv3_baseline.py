# -*- coding: utf-8 -*-
"""
FER System - MobileNetV3-Small 基线模型

纯净基线，不含任何注意力模块，用于独立训练作为蒸馏对比基准。
参考：https://pytorch.org/vision/stable/models/mobilenetv3.html
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from models.student_model_config import NUM_CLASSES


class MobileNetV3SmallBaseline(nn.Module):
    """MobileNetV3-Small 基线模型。

    纯基线结构：标准 MobileNetV3-Small backbone + 轻量分类头。
    无 SE/CBAM/ASPP 注意力，无知识蒸馏，仅作为对比基准。

    Architecture:
        - Backbone: torchvision.models.mobilenet_v3_small (ImageNet 预训练)
        - Features: [B, 3, H, W] → [B, 576, H', W'] (H'=H/32, W'=W/32)
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

        # ── Backbone ─────────────────────────────────────────────────────────
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # MobileNetV3-Small 特征层输出通道: [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576]
        # features: [B, 3, H, W] → [B, 576, H/32, W/32]
        self.features = backbone.features
        self.avgpool = backbone.avgpool  # GAP: [B, 576, 1, 1] → [B, 576, 1, 1]

        # ── 分类头 ──────────────────────────────────────────────────────────
        # 576 维经 GAP + Flatten → 576
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像 [B, 3, H, W]，支持 48/64/96/112 等任意尺寸。

        Returns:
            logits: [B, num_classes]，未经 softmax 的原始分数。
        """
        x = self.features(x)          # [B, 576, H/32, W/32]
        x = self.avgpool(x)           # [B, 576, 1, 1]
        x = self.classifier(x)        # [B, num_classes]
        return x

    @property
    def num_params(self) -> int:
        """返回模型可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MobileNetV3SmallBaseline(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 112, 112)

    with torch.no_grad():
        logits = model(x)

    print(f"模型参数: {model.num_params:,}")
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {logits.shape}")   # [2, 7]
    print(f"输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # 统计各层参数
    print("\n各层参数分布:")
    for name, module in model.named_modules():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n_params > 0 and "backbone" not in name:
            print(f"  {name:<35} {n_params:>10,} params  ({n_params/model.num_params*100:.1f}%)")

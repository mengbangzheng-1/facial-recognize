# -*- coding: utf-8 -*-
"""FER System - ConvNeXt-Base Teacher Model

Teacher model wrapper around torchvision's ConvNeXt-Base for
facial expression recognition with knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from models.student_model_config import NUM_CLASSES


class ConvNeXtTeacher(nn.Module):
    """ConvNeXt-Base teacher model for knowledge distillation.

    Wraps a pretrained ConvNeXt-Base with a modified classification head.
    Supports freezing the backbone for transfer learning.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = ConvNeXt_Base_Weights.DEFAULT
            self.backbone = convnext_base(weights=weights)
        else:
            self.backbone = convnext_base(weights=None)

        # Replace classifier head with Dropout for regularization
        # ConvNeXt classifier: features -> avgpool -> layer_norm -> flatten -> linear
        original_linear = self.backbone.classifier[-1]
        in_features = original_linear.in_features  # 1024 for Base
        
        # Replace linear with Dropout + Linear for regularization
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.3),  # 30% dropout to reduce overfitting
            nn.Linear(in_features, num_classes)
        )

        # Feature extraction hook
        self._features = None
        self.backbone.classifier[-2].register_forward_hook(self._feature_hook)

    def _feature_hook(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Hook to capture intermediate features before the final linear layer."""
        self._features = output

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x: Input image [B, 3, 112, 112].

        Returns:
            Tuple of (logits, features):
                - logits: [B, num_classes]
                - features: [B, feature_dim] intermediate features
        """
        logits = self.backbone(x)
        features = self._features
        return logits, features

    @classmethod
    def load_pretrained(cls, checkpoint_path: str) -> "ConvNeXtTeacher":
        """Load a teacher model from a checkpoint file.

        Args:
            checkpoint_path: Path to the saved model weights.

        Returns:
            ConvNeXtTeacher instance with loaded weights.
        """
        model = cls(pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

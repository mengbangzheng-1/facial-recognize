# -*- coding: utf-8 -*-
"""FER System - Loss Functions

FocalLoss, DistillationLoss, and CombinedLoss for knowledge distillation
training of the student model.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance with optional label smoothing.

    Down-weights well-classified examples and focuses on hard examples.
    Label smoothing helps prevent overconfidence and improves generalization.

    Args:
        alpha: Per-class weight tensor [num_classes]. If None, no class weighting.
        gamma: Focusing parameter. Higher values focus more on hard examples.
        reduction: Reduction mode ('mean', 'sum', or 'none').
        label_smoothing: Label smoothing factor (0.0 to 1.0).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model logits [B, num_classes].
            targets: Ground truth labels [B].

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha,
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DistillationLoss(nn.Module):
    """KL divergence distillation loss.

    Computes the KL divergence between softened student and teacher outputs.

    Args:
        temperature: Temperature for softmax softening.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            student_logits: Student model logits [B, num_classes].
            teacher_logits: Teacher model logits [B, num_classes].

        Returns:
            KL divergence loss scaled by T^2.
        """
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        return kl_loss * (self.temperature ** 2)


class CombinedLoss(nn.Module):
    """Combined Focal + KL distillation loss with optional label smoothing.

    Loss = focal_weight * FocalLoss + kl_weight * DistillationLoss

    Args:
        focal_weight: Weight for the focal loss component.
        kl_weight: Weight for the KL distillation component.
        temperature: Temperature for the distillation loss.
        alpha: Per-class weights for focal loss.
        gamma: Focusing parameter for focal loss.
        label_smoothing: Label smoothing factor for regularization.
    """

    def __init__(
        self,
        focal_weight: float = 0.7,
        kl_weight: float = 0.3,
        temperature: float = 4.0,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.kl_loss = DistillationLoss(temperature=temperature)
        self.focal_weight = focal_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            student_logits: Student model logits [B, num_classes].
            teacher_logits: Teacher model logits [B, num_classes].
            targets: Ground truth labels [B].

        Returns:
            Tuple of (total_loss, loss_info_dict).
        """
        focal = self.focal_loss(student_logits, targets)
        kl = self.kl_loss(student_logits, teacher_logits)
        total_loss = self.focal_weight * focal + self.kl_weight * kl

        loss_info = {
            "focal_loss": focal.item(),
            "kl_loss": kl.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_info

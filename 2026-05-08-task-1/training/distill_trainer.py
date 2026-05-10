# -*- coding: utf-8 -*-
"""FER System - Knowledge Distillation Trainer

Training loop for distilling knowledge from teacher to student model
using combined Focal + KL divergence loss.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.callbacks import EarlyStopping, ModelCheckpoint
from training.losses import CombinedLoss
from utils.logger import setup_logger


class DistillationTrainer:
    """Knowledge distillation training loop.

    Trains a student model using soft labels from a frozen teacher model
    with a combined Focal + KL divergence loss.

    Args:
        student: Student model to train.
        teacher: Teacher model (frozen during training).
        criterion: Combined loss function.
        optimizer: Optimizer for student parameters.
        device: Compute device.
        scheduler: Optional learning rate scheduler.
        checkpoint_dir: Directory for saving student model checkpoints.
        log_dir: Directory for TensorBoard logs.
        patience: Early stopping patience.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        criterion: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "model_checkpoints/student",
        log_dir: str = "logs",
        patience: int = 15,
    ):
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.early_stopping = EarlyStopping(patience=patience, mode="min")
        self.checkpoint = ModelCheckpoint(checkpoint_dir, mode="max")
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / "events"))
        self.logger = setup_logger("distill_trainer", log_dir)

        # Move models to device
        self.student.to(self.device)
        self.teacher.to(self.device)

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one distillation training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary with training metrics.
        """
        self.student.train()
        total_loss = 0.0
        total_focal = 0.0
        total_kl = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Student forward
            student_logits = self.student(images)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits, _ = self.teacher(images)

            # Compute combined loss
            loss, loss_info = self.criterion(student_logits, teacher_logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss_info["total_loss"]
            total_focal += loss_info["focal_loss"]
            total_kl += loss_info["kl_loss"]

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "loss": total_loss / len(train_loader),
            "focal": total_focal / len(train_loader),
            "kl": total_kl / len(train_loader),
            "acc": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary with validation metrics.
        """
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            student_logits = self.student(images)

            # Validation: only compute FocalLoss (no teacher needed for val metric)
            focal = self.criterion.focal_loss(student_logits, labels)
            total_loss += focal.item()

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "val_loss": total_loss / len(val_loader),
            "val_acc": 100.0 * correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, list]:
        """Run the full distillation training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.

        Returns:
            Dictionary of metric histories.
        """
        history = {
            "train_loss": [], "train_acc": [], "train_focal": [], "train_kl": [],
            "val_loss": [], "val_acc": [],
        }

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {train_metrics['loss']:.4f} "
                f"(Focal: {train_metrics['focal']:.4f}, KL: {train_metrics['kl']:.4f}) | "
                f"Train Acc: {train_metrics['acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            # TensorBoard logging
            self.writer.add_scalar("Distill/Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Distill/Focal/train", train_metrics["focal"], epoch)
            self.writer.add_scalar("Distill/KL/train", train_metrics["kl"], epoch)
            self.writer.add_scalar("Distill/Acc/train", train_metrics["acc"], epoch)
            self.writer.add_scalar("Distill/Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Distill/Acc/val", val_metrics["val_acc"], epoch)

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["train_focal"].append(train_metrics["focal"])
            history["train_kl"].append(train_metrics["kl"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])

            # Checkpoint - save based on val_acc (higher is better)
            self.checkpoint.step(val_metrics["val_acc"], self.student)

            # Early stopping
            if self.early_stopping.step(val_metrics["val_loss"]):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        return history

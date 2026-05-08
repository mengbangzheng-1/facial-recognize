# -*- coding: utf-8 -*-
"""FER System - Standard Trainer

Training loop with early stopping, model checkpointing, and TensorBoard logging.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.callbacks import EarlyStopping, ModelCheckpoint
from utils.logger import setup_logger


class Trainer:
    """Standard training loop for the teacher model.

    Handles training, validation, learning rate scheduling, early stopping,
    and checkpoint saving.

    Args:
        model: Neural network model.
        criterion: Loss function.
        optimizer: Parameter optimizer.
        device: Compute device.
        scheduler: Optional learning rate scheduler.
        checkpoint_dir: Directory for saving model checkpoints.
        log_dir: Directory for TensorBoard logs.
        patience: Early stopping patience.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints/teacher",
        log_dir: str = "logs",
        patience: int = 10,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.early_stopping = EarlyStopping(patience=patience, mode="min")
        self.checkpoint = ModelCheckpoint(checkpoint_dir, mode="min")
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / "events"))
        self.logger = setup_logger("trainer", log_dir)

        self.model.to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "train_loss": total_loss / len(train_loader),
            "train_acc": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = logits.max(1)
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
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.

        Returns:
            Dictionary of metric histories.
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics["train_acc"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["val_acc"], epoch)

            # Record history
            for key in history:
                if key in train_metrics:
                    history[key].append(train_metrics[key])
                elif key in val_metrics:
                    history[key].append(val_metrics[key])

            # Checkpoint
            self.checkpoint.step(val_metrics["val_loss"], self.model)

            # Early stopping
            if self.early_stopping.step(val_metrics["val_loss"]):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        return history

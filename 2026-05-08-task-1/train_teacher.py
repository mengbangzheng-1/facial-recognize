# -*- coding: utf-8 -*-
"""
FER System - ConvNeXt-Base Teacher Model Training (Single File Version)

Train command:
    python train_teacher.py --epochs 80 --batch-size 256 --lr 1e-4 --patience 20

Training result: ConvNeXt-Base teacher with 70.27% validation accuracy on FER2013.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


# ============================================================================
# Constants
# ============================================================================

EMOTION_LABELS: list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES: int = 7
IMAGE_SIZE: Tuple[int, int] = (112, 112)

# Training hyperparameters (matching the 70.27% run)
TEACHER_EPOCHS: int = 80
TEACHER_LR: float = 1e-4
TEACHER_WEIGHT_DECAY: float = 5e-4
BATCH_SIZE: int = 64
NUM_WORKERS: int = 4
PATIENCE: int = 20
FOCAL_GAMMA: float = 2.0
LABEL_SMOOTHING: float = 0.0
MAX_GRAD_NORM: float = 5.0


# ============================================================================
# Logging
# ============================================================================

def setup_logger(name: str = "fer_system", log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "training.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# ============================================================================
# Dataset
# ============================================================================

class FER2013Dataset(Dataset):
    """FER2013 facial expression dataset."""

    EMOTION_MAP: dict = {
        0: "angry", 1: "disgust", 2: "fear",
        3: "happy", 4: "sad", 5: "surprise", 6: "neutral",
    }

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        usage: Optional[str] = None,
    ):
        super().__init__()
        self.transform = transform

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if usage is not None:
            df = df[df["Usage"] == usage].reset_index(drop=True)

        self.data = []
        for _, row in df.iterrows():
            emotion = int(row["emotion"])
            pixels = np.array(row["pixels"].split(), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            self.data.append((image, emotion))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.data[idx]
        # Convert grayscale [48,48] to 3-channel [48,48,3]
        image = np.stack([image, image, image], axis=-1)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ============================================================================
# Data Transforms
# ============================================================================

def get_train_transforms(image_size: Tuple[int, int] = (112, 112)) -> transforms.Compose:
    """Training data augmentation pipeline."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0,
        ),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3),
        ),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (112, 112)) -> transforms.Compose:
    """Validation/test preprocessing pipeline."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Model
# ============================================================================

class ConvNeXtTeacher(nn.Module):
    """ConvNeXt-Base teacher model for knowledge distillation."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = ConvNeXt_Base_Weights.DEFAULT
            self.backbone = convnext_base(weights=weights)
        else:
            self.backbone = convnext_base(weights=None)

        # Replace classifier head: add Dropout for regularization
        original_linear = self.backbone.classifier[-1]
        in_features = original_linear.in_features  # 1024 for ConvNeXt-Base

        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        # Feature extraction hook
        self._features = None
        self.backbone.classifier[-2].register_forward_hook(self._feature_hook)

    def _feature_hook(self, module, input, output):
        self._features = output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.backbone(x)
        return logits, self._features


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# Callbacks
# ============================================================================

class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Model checkpoint callback to save the best model."""

    def __init__(self, save_dir: str, mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_score: Optional[float] = None

    def step(self, score: float, model: torch.nn.Module) -> bool:
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / "best_model.pth"
            try:
                torch.save(model.state_dict(), save_path)
                print(f"[ModelCheckpoint] Saved best model (acc={score:.2f}%) to {save_path}")
                return True
            except Exception as e:
                print(f"[ModelCheckpoint] Save failed: {e}")
                return False
        return False


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Standard training loop for the teacher model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "model_checkpoints/teacher",
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

        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.checkpoint = ModelCheckpoint(checkpoint_dir, mode="max")
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / "events"))
        self.logger = setup_logger("trainer", log_dir)

        self.model.to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.criterion(logits, labels)

            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)

        return {
            "train_loss": total_loss / len(train_loader),
            "train_acc": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
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
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics["train_acc"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["val_acc"], epoch)

            for key in history:
                if key in train_metrics:
                    history[key].append(train_metrics[key])
                elif key in val_metrics:
                    history[key].append(val_metrics[key])

            self.checkpoint.step(val_metrics["val_acc"], self.model)

            if self.early_stopping.step(val_metrics["val_acc"]):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        return history


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Base teacher model")
    parser.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TEACHER_LR)
    parser.add_argument("--weight-decay", type=float, default=TEACHER_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--data-dir", type=str, default="data/fer2013")
    parser.add_argument("--checkpoint-dir", type=str, default="model_checkpoints/teacher")
    parser.add_argument("--log-dir", type=str, default="logs/teacher")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_teacher", args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Image size: {IMAGE_SIZE}")

    # Dataset
    csv_path = f"{args.data_dir}/fer2013.csv"
    train_dataset = FER2013Dataset(
        csv_path, transform=get_train_transforms(IMAGE_SIZE), usage="Training"
    )
    val_dataset = FER2013Dataset(
        csv_path, transform=get_val_transforms(IMAGE_SIZE), usage="PublicTest"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = ConvNeXtTeacher(pretrained=True)

    # Discriminative learning rates
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_lr = args.lr * 0.5  # 5e-5 for pretrained backbone
    head_lr = args.lr            # 1e-4 for new head

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
    ])
    logger.info(
        f"ConvNeXt-Base: backbone LR={backbone_lr:.2e}, head LR={head_lr:.2e} "
        f"(discriminative LR)"
    )

    # Loss
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    logger.info(f"FocalLoss with label_smoothing={LABEL_SMOOTHING}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        patience=args.patience,
    )

    history = trainer.fit(train_loader, val_loader, args.epochs)
    logger.info(
        f"Training complete. Best val acc: {trainer.checkpoint.best_score:.2f}%"
    )


if __name__ == "__main__":
    main()

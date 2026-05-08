# -*- coding: utf-8 -*-
"""FER System - Teacher Model Training Script

Trains the ConvNeXt-Base teacher model on FER2013 dataset.
"""

import sys
import os

# 自动添加项目根目录到Python路径
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
_project_root = os.path.dirname(_script_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import FER2013Dataset
from data.transforms import get_train_transforms, get_val_transforms
from models.teacher_model import ConvNeXtTeacher
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.config import TrainConfig, get_device, TEACHER_CKPT_DIR
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Base teacher model")
    parser.add_argument("--epochs", type=int, default=TrainConfig.TEACHER_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=TrainConfig.TEACHER_LR,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.TEACHER_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=TrainConfig.TEACHER_PATIENCE,
                        help="Early stopping patience")
    parser.add_argument("--data-dir", type=str, default="data/fer2013",
                        help="FER2013 dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default=str(TEACHER_CKPT_DIR),
                        help="Checkpoint save directory")
    parser.add_argument("--log-dir", type=str, default="logs/teacher",
                        help="Log directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_teacher", args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Dataset
    csv_path = f"{args.data_dir}/fer2013.csv"
    train_dataset = FER2013Dataset(
        csv_path, transform=get_train_transforms(), usage="Training"
    )
    val_dataset = FER2013Dataset(
        csv_path, transform=get_val_transforms(), usage="PublicTest"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=TrainConfig.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = ConvNeXtTeacher(pretrained=True)
    model.freeze_backbone()
    logger.info("ConvNeXt-Base teacher model initialized with frozen backbone")

    # Loss and optimizer
    criterion = FocalLoss(gamma=TrainConfig.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=TrainConfig.SCHEDULER_FACTOR,
        patience=TrainConfig.SCHEDULER_PATIENCE,
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
    logger.info(f"Training complete. Best val loss: {trainer.checkpoint.best_score:.4f}")


if __name__ == "__main__":
    main()

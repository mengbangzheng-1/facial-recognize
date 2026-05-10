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
import math
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data.dataset import FER2013Dataset
from data.transforms import get_train_transforms, get_val_transforms
from models.teacher_model import ConvNeXtTeacher
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.config import TrainConfig, get_device, TEACHER_CKPT_DIR, IMAGE_SIZE
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Base teacher model")
    parser.add_argument("--epochs", type=int, default=TrainConfig.TEACHER_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=TrainConfig.TEACHER_LR,
                        help="Head learning rate (backbone gets 1% of this)")
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
    parser.add_argument("--no-mixup", action="store_true", default=True,
                        help="Disable Mixup (default: disabled)")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable Mixup")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="Mixup alpha parameter")
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_teacher", args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Image size: {IMAGE_SIZE}")

    # Dataset with enhanced augmentation and 64x64 resolution
    csv_path = f"{args.data_dir}/fer2013.csv"
    train_dataset = FER2013Dataset(
        csv_path, transform=get_train_transforms(IMAGE_SIZE), usage="Training"
    )
    val_dataset = FER2013Dataset(
        csv_path, transform=get_val_transforms(IMAGE_SIZE), usage="PublicTest"
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

    # Discriminative learning rates: small LR for backbone, large LR for head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_lr = args.lr * 0.5  # 5e-5 for pretrained backbone (moderate adaptation)
    head_lr = args.lr              # 1e-4 for new head

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
        ]
    )
    logger.info(f"ConvNeXt-Base: backbone LR={backbone_lr:.2e}, head LR={head_lr:.2e} (discriminative LR)")

    # Loss with Label Smoothing
    criterion = FocalLoss(
        gamma=TrainConfig.FOCAL_GAMMA,
        label_smoothing=TrainConfig.LABEL_SMOOTHING
    )
    logger.info(f"FocalLoss with label_smoothing={TrainConfig.LABEL_SMOOTHING}")

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
        use_mixup=args.mixup and not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
    )
    logger.info(f"Mixup: {'enabled' if args.mixup else 'disabled'} (alpha={args.mixup_alpha})")

    history = trainer.fit(train_loader, val_loader, args.epochs)
    logger.info(f"Training complete. Best val acc: {trainer.checkpoint.best_score:.4f}%")


if __name__ == "__main__":
    main()

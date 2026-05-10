# -*- coding: utf-8 -*-
"""
FER System - MobileNetV3-Small 基线训练脚本

纯净基线模型独立训练，用于蒸馏前的独立基准对比。
训练完成后，可作为 ImprovedMobileNetV3Small 蒸馏训练的目标基准。
"""

import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import torch
from torch.utils.data import DataLoader

from data.dataset import FER2013Dataset
from data.transforms import get_train_transforms, get_val_transforms
from models.mobilenetv3_baseline import MobileNetV3SmallBaseline
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.config import TrainConfig, get_device, IMAGE_SIZE
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="data/fer2013")
    parser.add_argument("--checkpoint-dir", type=str, default="model_checkpoints/baseline")
    parser.add_argument("--log-dir", type=str, default="logs/baseline")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_baseline", args.log_dir)

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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=TrainConfig.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = MobileNetV3SmallBaseline(num_classes=7, pretrained=True)
    num_params = model.num_params
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer - discriminative LR
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay},
        {"params": classifier_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])

    # Loss
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        patience=args.patience,
        use_mixup=False,
    )

    history = trainer.fit(train_loader, val_loader, args.epochs)
    logger.info(f"Baseline training complete. Best val acc: {trainer.checkpoint.best_score:.2f}%")


if __name__ == "__main__":
    main()

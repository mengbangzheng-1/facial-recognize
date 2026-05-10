# -*- coding: utf-8 -*-
"""
FER System - MobileNetV3-Large 基线训练脚本

纯净 MobileNetV3-Large 基线独立训练。
参数量约 5.5M，比 Small 基线高 4-6% 准确率，但模型大 5 倍。
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
from models.mobilenetv3_large_baseline import MobileNetV3LargeBaseline
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.config import TrainConfig, get_device, IMAGE_SIZE
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="data/fer2013")
    parser.add_argument("--checkpoint-dir", type=str, default="model_checkpoints/large_baseline")
    parser.add_argument("--log-dir", type=str, default="logs/large_baseline")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_large_baseline", args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Image size: {IMAGE_SIZE}")

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

    model = MobileNetV3LargeBaseline(num_classes=7, pretrained=True)
    logger.info(f"Model parameters: {model.num_params:,}")

    # Discriminative LR: backbone 0.1× head LR
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

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

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
    logger.info(f"Large baseline training complete. Best val acc: {trainer.checkpoint.best_score:.2f}%")


if __name__ == "__main__":
    main()

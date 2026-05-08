# -*- coding: utf-8 -*-
"""FER System - Student Model Distillation Training Script

Trains the improved MobileNetV3-Small student model using knowledge
distillation from a pretrained ConvNeXt-Base teacher model.
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

from data.dataset import FER2013Dataset
from data.transforms import get_train_transforms, get_val_transforms
from models.student_model import ImprovedMobileNetV3Small
from models.teacher_model import ConvNeXtTeacher
from training.distill_trainer import DistillationTrainer
from training.losses import CombinedLoss
from utils.config import TrainConfig, get_device, STUDENT_CKPT_DIR, IMAGE_SIZE
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train student model with knowledge distillation"
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.STUDENT_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=TrainConfig.STUDENT_LR,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.STUDENT_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=TrainConfig.STUDENT_PATIENCE,
                        help="Early stopping patience")
    parser.add_argument("--temperature", type=float, default=TrainConfig.DISTILL_TEMPERATURE,
                        help="Distillation temperature")
    parser.add_argument("--focal-weight", type=float, default=TrainConfig.FOCAL_LOSS_WEIGHT,
                        help="Focal loss weight")
    parser.add_argument("--kl-weight", type=float, default=TrainConfig.KL_LOSS_WEIGHT,
                        help="KL distillation loss weight")
    parser.add_argument("--teacher-path", type=str,
                        default="model_checkpoints/teacher/best_model.pth",
                        help="Path to trained teacher model weights")
    parser.add_argument("--data-dir", type=str, default="data/fer2013",
                        help="FER2013 dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default=str(STUDENT_CKPT_DIR),
                        help="Checkpoint save directory")
    parser.add_argument("--log-dir", type=str, default="logs/student",
                        help="Log directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    """Main distillation training entry point."""
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_student", args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Temperature: {args.temperature}, Focal: {args.focal_weight}, KL: {args.kl_weight}")
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

    # Teacher model
    logger.info(f"Loading teacher model from: {args.teacher_path}")
    teacher = ConvNeXtTeacher.load_pretrained(args.teacher_path)
    logger.info("Teacher model loaded successfully")

    # Student model
    student = ImprovedMobileNetV3Small(num_classes=7)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student model parameters: {total_params:,}")

    # Combined loss with label smoothing
    criterion = CombinedLoss(
        focal_weight=args.focal_weight,
        kl_weight=args.kl_weight,
        temperature=args.temperature,
        gamma=TrainConfig.FOCAL_GAMMA,
        label_smoothing=TrainConfig.LABEL_SMOOTHING,
    )
    logger.info(f"CombinedLoss with label_smoothing={TrainConfig.LABEL_SMOOTHING}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=TrainConfig.SCHEDULER_FACTOR,
        patience=TrainConfig.SCHEDULER_PATIENCE,
    )

    # Train
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        patience=args.patience,
    )

    history = trainer.fit(train_loader, val_loader, args.epochs)
    logger.info(f"Distillation training complete. Best val loss: {trainer.checkpoint.best_score:.4f}")


if __name__ == "__main__":
    main()

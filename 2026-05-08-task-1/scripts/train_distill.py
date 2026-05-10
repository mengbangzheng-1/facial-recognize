# -*- coding: utf-8 -*-
"""FER System - Knowledge Distillation Training Script

Trains ImprovedMobileNetV3Small with:
  ✅ SE attention
  ✅ CBAM attention
  ✅ ASPP multi-scale fusion
  ✅ Focal Loss
  ✅ Data augmentation (via dataset transforms)
  ✅ Cosine annealing scheduler
  ✅ Knowledge distillation from ConvNeXt-Base teacher (70.27% val_acc)

Hyperparameters match the baseline (train_baseline.py) exactly:
  lr=1e-4, weight_decay=1e-4, batch_size=256, epochs=100, patience=20
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
from models.teacher_model import ConvNeXtTeacher
from models.student_model import ImprovedMobileNetV3Small
from training.losses import CombinedLoss
from training.distill_trainer import DistillationTrainer
from utils.config import TrainConfig, get_device, IMAGE_SIZE
from utils.logger import setup_logger


def load_teacher(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load ConvNeXt-Base teacher model from checkpoint.

    Supports three checkpoint formats:
      - dict with 'model_state_dict' key
      - dict with 'state_dict' key
      - raw state_dict

    Args:
        ckpt_path: Path to checkpoint file.
        device: Target device.

    Returns:
        ConvNeXtTeacher model in eval mode with frozen parameters.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Please train the teacher first: python scripts/train_teacher.py"
        )

    model = ConvNeXtTeacher(pretrained=False)
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # CRITICAL: move model to device to avoid CPU/CUDA mismatch
    model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Sanity check: dummy forward
    with torch.no_grad():
        dummy = torch.randn(2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
        logits, feats = model(dummy)
        print(f"[Teacher] logits={logits.shape}, feats={feats.shape}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: ImprovedMobileNetV3Small + ConvNeXt-Base teacher"
    )
    # ── Same hyperparams as baseline ────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    # ── Distillation specific ────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--focal-weight", type=float, default=0.3)
    parser.add_argument("--kl-weight", type=float, default=0.7)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    # ── Paths ───────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=str, default="data/fer2013")
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="model_checkpoints/teacher/best_teacher.pth",
        help="Path to ConvNeXt-Base teacher checkpoint (70.27%% val_acc)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="model_checkpoints/student",
        help="Directory to save student checkpoints",
    )
    parser.add_argument("--log-dir", type=str, default="logs/distill")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)
    logger = setup_logger("train_distill", args.log_dir)

    logger.info("=" * 60)
    logger.info("Knowledge Distillation Training")
    logger.info(f"  Student  : ImprovedMobileNetV3Small (+SE/CBAM/ASPP)")
    logger.info(f"  Teacher  : ConvNeXt-Base (70.27%% val_acc)")
    logger.info(f"  Image    : {IMAGE_SIZE}")
    logger.info(f"  Epochs   : {args.epochs}")
    logger.info(f"  Batch    : {args.batch_size}")
    logger.info(f"  LR       : {args.lr} (backbone x0.5 = {args.lr * 0.5:.2e})")
    logger.info(f"  WD       : {args.weight_decay}")
    logger.info(f"  Patience : {args.patience}")
    logger.info(f"  T (temp) : {args.temperature}")
    logger.info(f"  Loss     : {args.focal_weight}*Focal + {args.kl_weight}*KL")
    logger.info(f"  Teacher ckpt: {args.teacher_ckpt}")
    logger.info("=" * 60)

    # ── Dataset ────────────────────────────────────────────────────
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
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Teacher (frozen) ───────────────────────────────────────────
    logger.info(f"Loading teacher from {args.teacher_ckpt} ...")
    teacher = load_teacher(args.teacher_ckpt, device)
    logger.info("Teacher loaded and frozen.")

    # ── Student ─────────────────────────────────────────────────────
    student = ImprovedMobileNetV3Small(num_classes=7, pretrained=True)
    n_params = student.num_params
    logger.info(f"Student parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── Optimizer (discriminative LR, same as baseline) ────────────
    backbone_params, classifier_params = [], []
    for name, param in student.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.5, "weight_decay": args.weight_decay},
        {"params": classifier_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])
    logger.info(
        f"Optimizer: backbone LR={args.lr * 0.5:.2e}, "
        f"classifier LR={args.lr:.2e} (discriminative)"
    )

    # ── Combined Loss (Focal + KL) ─────────────────────────────────
    criterion = CombinedLoss(
        focal_weight=args.focal_weight,
        kl_weight=args.kl_weight,
        temperature=args.temperature,
        gamma=args.focal_gamma,
        label_smoothing=0.05,
    )
    logger.info(
        f"CombinedLoss: focal_weight={args.focal_weight}, "
        f"kl_weight={args.kl_weight}, T={args.temperature}, "
        f"gamma={args.focal_gamma}"
    )

    # ── Scheduler: Cosine Annealing (same as baseline) ────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # ── Distillation Trainer ────────────────────────────────────────
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

    # ── Train ───────────────────────────────────────────────────────
    history = trainer.fit(train_loader, val_loader, args.epochs)

    best_acc = trainer.checkpoint.best_score
    logger.info("=" * 60)
    logger.info(f"Distillation complete!")
    logger.info(f"  Best student val acc: {best_acc:.2f}%")
    logger.info(f"  Teacher val acc    : 70.27%")
    logger.info(f"  Gap                 : {best_acc - 70.27:+.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

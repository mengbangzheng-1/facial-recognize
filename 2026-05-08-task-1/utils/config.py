# -*- coding: utf-8 -*-
"""FER System - Configuration Management

Project-wide hyperparameters, path constants, and device management.
"""

from pathlib import Path
from typing import Optional

import torch


# ===========================================================================
# Path Constants
# ===========================================================================
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data" / "fer2013"
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
LOG_DIR: Path = ROOT_DIR / "logs"

TEACHER_CKPT_DIR: Path = CHECKPOINT_DIR / "teacher"
STUDENT_CKPT_DIR: Path = CHECKPOINT_DIR / "student"


# ===========================================================================
# Emotion Labels
# ===========================================================================
EMOTION_CLASSES: list = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]
NUM_CLASSES: int = 7


# ===========================================================================
# Training Hyperparameters
# ===========================================================================
class TrainConfig:
    """Training hyperparameters for teacher and student models."""

    # Teacher model
    TEACHER_EPOCHS: int = 50
    TEACHER_LR: float = 1e-4
    TEACHER_WEIGHT_DECAY: float = 1e-4

    # Student model
    STUDENT_EPOCHS: int = 100
    STUDENT_LR: float = 3e-4
    STUDENT_WEIGHT_DECAY: float = 1e-4

    # Common
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4

    # Learning rate scheduler
    SCHEDULER_PATIENCE: int = 5
    SCHEDULER_FACTOR: float = 0.5

    # Early stopping
    TEACHER_PATIENCE: int = 10
    STUDENT_PATIENCE: int = 15

    # Distillation
    DISTILL_TEMPERATURE: float = 4.0
    FOCAL_LOSS_WEIGHT: float = 0.7
    KL_LOSS_WEIGHT: float = 0.3
    FOCAL_GAMMA: float = 2.0

    # Gradient clipping
    MAX_GRAD_NORM: float = 5.0


# ===========================================================================
# Device Management
# ===========================================================================
def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available compute device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available.

    Returns:
        torch.device instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ===========================================================================
# Exception Definitions
# ===========================================================================
class FERSystemError(Exception):
    """FER system base exception."""
    pass


class ModelLoadError(FERSystemError):
    """Model loading failed."""
    pass


class FaceDetectionError(FERSystemError):
    """Face detection failed."""
    pass


class DataLoadError(FERSystemError):
    """Data loading failed."""
    pass

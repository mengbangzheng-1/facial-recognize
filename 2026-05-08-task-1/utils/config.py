# -*- coding: utf-8 -*-
"""FER System - Configuration Management

Project-wide hyperparameters, path constants, and device management.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch


# ===========================================================================
# Path Constants
# ===========================================================================
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data" / "fer2013"
CHECKPOINT_DIR: Path = ROOT_DIR / "model_checkpoints"
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
# Image Configuration
# ===========================================================================
IMAGE_SIZE: Tuple[int, int] = (64, 64)  # Increased from 48x48 for better accuracy


# ===========================================================================
# Training Hyperparameters
# ===========================================================================
class TrainConfig:
    """Training hyperparameters for teacher and student models."""

    # Teacher model - optimized for FER2013
    TEACHER_EPOCHS: int = 50
    TEACHER_LR: float = 5e-5  # Reduced from 1e-4 for better convergence
    TEACHER_WEIGHT_DECAY: float = 1e-4

    # Student model
    STUDENT_EPOCHS: int = 100
    STUDENT_LR: float = 1e-4  # Reduced for stability
    STUDENT_WEIGHT_DECAY: float = 1e-4

    # Common
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4

    # Learning rate scheduler - cosine annealing for better convergence
    SCHEDULER_PATIENCE: int = 5
    SCHEDULER_FACTOR: float = 0.5

    # Early stopping - more patient for larger models
    TEACHER_PATIENCE: int = 25  # Increased for longer training
    STUDENT_PATIENCE: int = 20

    # Distillation
    DISTILL_TEMPERATURE: float = 4.0
    FOCAL_LOSS_WEIGHT: float = 0.7
    KL_LOSS_WEIGHT: float = 0.3
    FOCAL_GAMMA: float = 2.0

    # Label smoothing for regularization
    LABEL_SMOOTHING: float = 0.1

    # Gradient clipping
    MAX_GRAD_NORM: float = 1.0  # Reduced for stability


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

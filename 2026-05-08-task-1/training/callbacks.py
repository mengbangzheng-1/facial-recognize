# -*- coding: utf-8 -*-
"""FER System - Training Callbacks

EarlyStopping, ModelCheckpoint, and learning rate scheduling callbacks.
"""

from pathlib import Path
from typing import Optional

import torch


class EarlyStopping:
    """Early stopping callback to halt training when validation metric stalls.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        """Check if the current score is an improvement."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float) -> bool:
        """Check whether training should stop.

        Args:
            score: Current epoch's validation metric.

        Returns:
            True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Model checkpoint callback to save the best model weights.

    Args:
        save_dir: Directory to save checkpoints.
        mode: 'min' for loss, 'max' for accuracy.
    """

    def __init__(self, save_dir: str, mode: str = "min"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_score: Optional[float] = None

    def step(self, score: float, model: torch.nn.Module) -> bool:
        """Check and save model if current score is the best.

        Args:
            score: Current validation metric.
            model: Model to save.

        Returns:
            True if the model was saved (new best).
        """
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            save_path = self.save_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            return True
        return False

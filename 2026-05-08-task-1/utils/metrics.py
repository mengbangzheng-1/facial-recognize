# -*- coding: utf-8 -*-
"""FER System - Utility: Evaluation Metrics

Precision, recall, F1, and accuracy computation utilities.
"""

from typing import Dict, List

import numpy as np


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted labels [N].
        labels: Ground truth labels [N].

    Returns:
        Accuracy as a percentage.
    """
    return float((predictions == labels).mean() * 100)


def compute_per_class_accuracy(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[int, float]:
    """Compute per-class accuracy.

    Args:
        predictions: Predicted labels [N].
        labels: Ground truth labels [N].
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping class index to accuracy percentage.
    """
    result = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[c] = 0.0
        else:
            result[c] = float((predictions[mask] == c).mean() * 100)
    return result


def compute_confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: Predicted labels [N].
        labels: Ground truth labels [N].
        num_classes: Total number of classes.

    Returns:
        Confusion matrix [num_classes, num_classes].
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, predictions):
        cm[int(t), int(p)] += 1
    return cm

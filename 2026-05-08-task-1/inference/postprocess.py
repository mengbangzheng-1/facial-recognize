# -*- coding: utf-8 -*-
"""FER System - Inference Post-processing

Probability normalization and result formatting utilities.
"""

from typing import Dict, List, Tuple

import numpy as np


EMOTION_LABELS: list = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]


def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize a probability dictionary to sum to 1.0.

    Args:
        probs: Dictionary of {emotion: probability}.

    Returns:
        Normalized probability dictionary.
    """
    total = sum(probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}


def get_top_k(
    probs: Dict[str, float], k: int = 3
) -> List[Tuple[str, float]]:
    """Get the top-K emotions sorted by probability.

    Args:
        probs: Dictionary of {emotion: probability}.
        k: Number of top results to return.

    Returns:
        List of (emotion, probability) tuples sorted descending.
    """
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:k]


def format_detection_result(
    bbox: Tuple[int, int, int, int],
    emotions: Dict[str, float],
) -> Dict:
    """Format a single face detection result.

    Args:
        bbox: Face bounding box (x, y, w, h).
        emotions: Emotion probability dictionary.

    Returns:
        Formatted result dictionary.
    """
    top_emotion = max(emotions, key=emotions.get)
    top_confidence = emotions[top_emotion]
    return {
        "bbox": bbox,
        "emotions": emotions,
        "top_emotion": top_emotion,
        "top_confidence": top_confidence,
    }

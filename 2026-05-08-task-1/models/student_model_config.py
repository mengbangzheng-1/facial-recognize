# -*- coding: utf-8 -*-
"""FER System - Student Model Configuration Constants

Shared constants for the improved MobileNetV3-Small student model.
"""

# Expression categories
EMOTION_LABELS: list = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]
NUM_CLASSES: int = 7

# MobileNetV3-Small channel configuration per InvertedResidual block
MOBILENETV3_SMALL_CHANNELS: list = [16, 16, 24, 40, 40, 40, 48, 48, 96, 96, 96]

# SE module reduction ratio
SE_REDUCTION_RATIO: int = 16

# CBAM configuration
CBAM_CHANNEL_REDUCTION: int = 16
CBAM_SPATIAL_KERNEL: int = 7

# ASPP configuration
ASPP_DILATIONS: list = [1, 6, 12, 18]
ASPP_OUT_CHANNELS: int = 96

# Classifier head configuration
CLASSIFIER_HIDDEN_DIM: int = 256
DROPOUT_FIRST: float = 0.2
DROPOUT_FINAL: float = 0.5

# -*- coding: utf-8 -*-
"""FER System - Data Transforms

Image augmentation and preprocessing pipelines for FER2013 dataset.
"""

from typing import Tuple

import torch
from torchvision import transforms


def get_train_transforms(image_size: Tuple[int, int] = (48, 48)) -> transforms.Compose:
    """Create training data augmentation pipeline.

    Args:
        image_size: Target image size (H, W).

    Returns:
        Composed transform for training.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (48, 48)) -> transforms.Compose:
    """Create validation/test preprocessing pipeline.

    Args:
        image_size: Target image size (H, W).

    Returns:
        Composed transform for validation/testing.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

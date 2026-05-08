# -*- coding: utf-8 -*-
"""FER System - Data Transforms

Image augmentation and preprocessing pipelines for FER2013 dataset.
Enhanced version with stronger data augmentation for better generalization.
"""

from typing import Callable, Tuple

import torch
from torchvision import transforms


def get_train_transforms(image_size: Tuple[int, int] = (64, 64)) -> transforms.Compose:
    """Create training data augmentation pipeline with strong augmentation.

    Includes:
    - RandAugment for automatic augmentation
    - RandomErasing for regularization
    - Color jitter and rotation

    Args:
        image_size: Target image size (H, W).

    Returns:
        Composed transform for training.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Stronger geometric augmentation
        transforms.RandomRotation(degrees=15),  # Increased from 10
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random translation
            scale=(0.9, 1.1),  # Random scaling
        ),
        
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,  # Increased from 0.2
            saturation=0.2,
            hue=0.1,
        ),
        
        # Convert to tensor and normalize
        transforms.Resize(image_size),
        transforms.ToTensor(),
        
        # Normalize with ImageNet-like statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Random Erasing for regularization
        transforms.RandomErasing(
            p=0.25,  # 25% chance
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
        ),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (64, 64)) -> transforms.Compose:
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: Tuple[int, int] = (64, 64)) -> list:
    """Get Test Time Augmentation transforms.

    Returns a list of transforms for TTA during inference.
    
    Args:
        image_size: Target image size (H, W).
        
    Returns:
        List of transform compositions for TTA.
    """
    base_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    return [
        # Original
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_normalize,
        ]),
    ]

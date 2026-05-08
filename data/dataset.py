# -*- coding: utf-8 -*-
"""FER System - FER2013 Dataset Loader

Loads and parses FER2013 CSV dataset for facial expression recognition.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """FER2013 facial expression dataset.

    Reads the Kaggle FER2013 CSV file with columns: emotion, pixels, Usage.

    Args:
        csv_path: Path to fer2013.csv file.
        transform: Optional image transform callable.
        usage: Filter by Usage column. One of 'Training', 'PublicTest',
            'PrivateTest', or None for all data.
    """

    EMOTION_MAP: dict = {
        0: "angry", 1: "disgust", 2: "fear",
        3: "happy", 4: "sad", 5: "surprise", 6: "neutral",
    }

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        usage: Optional[str] = None,
    ):
        super().__init__()
        self.transform = transform

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if usage is not None:
            df = df[df["Usage"] == usage].reset_index(drop=True)

        self.data = []
        for _, row in df.iterrows():
            emotion = int(row["emotion"])
            pixels = np.array(row["pixels"].split(), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            self.data.append((image, emotion))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        image, label = self.data[idx]

        # Convert grayscale to 3-channel for model compatibility
        image = np.stack([image, image, image], axis=-1)  # [48, 48, 3]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

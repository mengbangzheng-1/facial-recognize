# -*- coding: utf-8 -*-
"""FER System - Face Detector

OpenCV Haar cascade-based face detection for real-time inference.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class FaceDetector:
    """OpenCV Haar cascade face detector.

    Uses the built-in frontal face Haar cascade classifier.
    No external model files required.

    Args:
        scale_factor: Scale factor for multi-scale detection.
        min_neighbors: Minimum neighbors for detection filtering.
        min_size: Minimum face size (width, height).
    """

    HAAR_CASCADE_PATH: str = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)

        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {self.HAAR_CASCADE_PATH}"
            )

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect all faces in an image.

        Args:
            image: BGR format image [H, W, 3].

        Returns:
            List of face bounding boxes [(x, y, w, h), ...].
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if len(faces) == 0:
            return []
        return [tuple(int(v) for v in f) for f in faces]

    @staticmethod
    def crop_face(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (48, 48),
        expand_ratio: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Crop and resize a face region from the image.

        Args:
            image: BGR format image [H, W, 3].
            bbox: Face bounding box (x, y, w, h).
            target_size: Output size (width, height).
            expand_ratio: Ratio to expand the bounding box.

        Returns:
            Cropped and resized face image, or None if crop fails.
        """
        x, y, w, h = bbox

        # Expand bounding box
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h)
        x2 = min(image.shape[1], x + w + expand_w)
        y2 = min(image.shape[0], y + h + expand_h)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        return face

# -*- coding: utf-8 -*-
"""FER System - Display Widget

Left-side widget for displaying camera feed with detection overlays.
"""

from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.utils import ndarray_to_qimage


# Emotion colors for bounding boxes
EMOTION_COLORS: dict = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 140, 255),  # Orange
    "fear": (128, 0, 128),     # Purple
    "happy": (0, 255, 0),      # Green
    "sad": (255, 0, 0),        # Blue
    "surprise": (0, 255, 255), # Yellow
    "neutral": (200, 200, 200),# Gray
}


class DisplayWidget(QWidget):
    """Widget for displaying camera/video frames with detection overlays.

    Renders frames and draws bounding boxes with emotion labels
    directly on the image before display.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setStyleSheet("background-color: #1a1a2e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)

        self._current_frame: np.ndarray = np.zeros((360, 480, 3), dtype=np.uint8)

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the displayed frame.

        Args:
            frame: BGR image [H, W, 3].
        """
        self._current_frame = frame.copy()
        self._display_image(frame)

    def draw_results(self, results: List[dict]) -> None:
        """Draw detection results on the current frame.

        Args:
            results: List of detection result dictionaries with
                'bbox', 'emotions', and 'top_emotion' keys.
        """
        frame = self._current_frame.copy()

        for result in results:
            x, y, w, h = result["bbox"]
            emotion = result["top_emotion"]
            confidence = result["emotions"][emotion]

            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label = f"{emotion}: {confidence:.1%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame,
                (x, y - label_h - 8),
                (x + label_w, y),
                color,
                -1,
            )
            cv2.putText(
                frame, label,
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        self._current_frame = frame
        self._display_image(frame)

    def _display_image(self, image: np.ndarray) -> None:
        """Convert and display a numpy image.

        Args:
            image: BGR image [H, W, 3].
        """
        qimg = ndarray_to_qimage(image)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def clear(self) -> None:
        """Clear the display."""
        self.image_label.clear()

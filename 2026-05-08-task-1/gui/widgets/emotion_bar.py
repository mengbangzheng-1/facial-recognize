# -*- coding: utf-8 -*-
"""FER System - Emotion Bar Widget

Horizontal bar chart widget for displaying emotion probabilities.
"""

from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from models.student_model_config import EMOTION_LABELS


# Bar colors for each emotion
BAR_COLORS: dict = {
    "angry": QColor(220, 50, 50),
    "disgust": QColor(255, 140, 0),
    "fear": QColor(148, 50, 180),
    "happy": QColor(50, 200, 50),
    "sad": QColor(50, 100, 220),
    "surprise": QColor(255, 220, 0),
    "neutral": QColor(160, 160, 160),
}


class EmotionBarWidget(QWidget):
    """Custom widget displaying emotion probabilities as horizontal bars.

    Each emotion gets a labeled horizontal bar whose width represents
    the probability value.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._probs: Dict[str, float] = {e: 0.0 for e in EMOTION_LABELS}

    def update_probabilities(self, probs: Dict[str, float]) -> None:
        """Update the displayed probabilities.

        Args:
            probs: Dictionary mapping emotion names to probabilities.
        """
        self._probs = probs
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the emotion bars."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_height = max((h - 20) // len(EMOTION_LABELS) - 4, 16)
        label_width = 70
        bar_max_width = w - label_width - 60
        y_offset = 10

        for i, emotion in enumerate(EMOTION_LABELS):
            y = y_offset + i * (bar_height + 4)
            prob = self._probs.get(emotion, 0.0)
            bar_width = int(prob * bar_max_width)
            color = BAR_COLORS.get(emotion, QColor(200, 200, 200))

            # Draw emotion label
            painter.setPen(QPen(Qt.white))
            painter.drawText(0, y, label_width, bar_height, Qt.AlignRight | Qt.AlignVCenter, emotion)

            # Draw bar background
            painter.fillRect(label_width + 5, y, bar_max_width, bar_height, QColor(40, 40, 40))

            # Draw bar fill
            painter.fillRect(label_width + 5, y, bar_width, bar_height, color)

            # Draw percentage text
            painter.drawText(
                label_width + bar_max_width + 10, y,
                50, bar_height,
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{prob:.0%}",
            )

        painter.end()

    def clear(self) -> None:
        """Reset all probabilities to zero."""
        self._probs = {e: 0.0 for e in EMOTION_LABELS}
        self.update()

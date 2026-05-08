# -*- coding: utf-8 -*-
"""FER System - Result Overlay Widget

Overlay widget for displaying detection result details.
"""

from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class ResultOverlay(QWidget):
    """Widget for displaying detailed detection results.

    Shows the top emotion and all probability values for the
    most recently detected face.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.title_label = QLabel("Detection Result")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.title_label)

        self.result_label = QLabel("No detection yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.result_label)

        layout.addStretch()

    def update_result(self, result: Dict) -> None:
        """Update the displayed detection result.

        Args:
            result: Detection result dictionary with 'top_emotion',
                'top_confidence', and 'emotions' keys.
        """
        if result is None:
            self.result_label.setText("No face detected")
            return

        top = result.get("top_emotion", "N/A")
        conf = result.get("top_confidence", 0.0)
        emotions = result.get("emotions", {})

        lines = [f"Top: {top} ({conf:.1%})", ""]
        for emotion, prob in sorted(emotions.items(), key=lambda x: -x[1]):
            lines.append(f"  {emotion}: {prob:.1%}")

        self.result_label.setText("\n".join(lines))

    def clear(self) -> None:
        """Clear the display."""
        self.result_label.setText("No detection yet")

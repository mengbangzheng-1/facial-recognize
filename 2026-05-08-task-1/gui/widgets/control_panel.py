# -*- coding: utf-8 -*-
"""FER System - Control Panel Widget

Right-side control panel with buttons, statistics, and emotion display.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from gui.widgets.emotion_bar import EmotionBarWidget
from gui.widgets.result_overlay import ResultOverlay
from models.student_model_config import EMOTION_LABELS


class ControlPanel(QWidget):
    """Right-side control panel for the FER system GUI.

    Contains camera control buttons, emotion probability bars,
    detection result overlay, statistics, and FPS display.

    Signals:
        start_clicked: Camera start button clicked.
        stop_clicked: Camera stop button clicked.
        image_loaded: Image file selected for inference.
        video_loaded: Video file selected for inference.
    """

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    image_loaded = pyqtSignal(str)
    video_loaded = pyqtSignal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMaximumWidth(360)
        self.setMinimumWidth(280)

        self._emotion_counts: Counter = Counter()
        self._total_frames: int = 0

        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Control Buttons ---
        btn_group = QGroupBox("Controls")
        btn_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setStyleSheet("padding: 8px; font-size: 13px;")
        self.start_btn.clicked.connect(self.start_clicked.emit)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("padding: 8px; font-size: 13px;")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.stop_btn.setEnabled(False)

        self.load_image_btn = QPushButton("Import Image")
        self.load_image_btn.setStyleSheet("padding: 6px; font-size: 12px;")
        self.load_image_btn.clicked.connect(self._on_load_image)

        self.load_video_btn = QPushButton("Import Video")
        self.load_video_btn.setStyleSheet("padding: 6px; font-size: 12px;")
        self.load_video_btn.clicked.connect(self._on_load_video)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_layout.addLayout(btn_row)

        file_row = QHBoxLayout()
        file_row.addWidget(self.load_image_btn)
        file_row.addWidget(self.load_video_btn)
        btn_layout.addLayout(file_row)

        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        # --- Emotion Bars ---
        bar_group = QGroupBox("Emotion Probabilities")
        bar_layout = QVBoxLayout()
        self.emotion_bar = EmotionBarWidget()
        bar_layout.addWidget(self.emotion_bar)
        bar_group.setLayout(bar_layout)
        layout.addWidget(bar_group)

        # --- Detection Result ---
        self.result_overlay = ResultOverlay()
        layout.addWidget(self.result_overlay)

        # --- Statistics ---
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("Total frames: 0\nNo detections yet")
        self.stats_label.setStyleSheet("font-size: 12px;")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # --- FPS ---
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(self.fps_label)

        # --- Bottom Buttons ---
        bottom_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Log")
        self.save_btn.setStyleSheet("padding: 6px;")
        self.save_btn.clicked.connect(self._on_save_log)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("padding: 6px;")
        self.clear_btn.clicked.connect(self._on_clear)

        bottom_row.addWidget(self.save_btn)
        bottom_row.addWidget(self.clear_btn)
        layout.addLayout(bottom_row)

        layout.addStretch()

    def set_running(self, running: bool) -> None:
        """Toggle button enabled states based on running status.

        Args:
            running: Whether the camera is currently running.
        """
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def update_emotions(self, emotions: Dict[str, float]) -> None:
        """Update the emotion probability bars.

        Args:
            emotions: Dictionary mapping emotion names to probabilities.
        """
        self.emotion_bar.update_probabilities(emotions)

    def update_result(self, result: Optional[Dict]) -> None:
        """Update the detection result overlay.

        Args:
            result: Detection result dictionary or None.
        """
        self.result_overlay.update_result(result)

    def update_fps(self, fps: float) -> None:
        """Update the FPS display.

        Args:
            fps: Current frames per second.
        """
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_statistics(self, results: list) -> None:
        """Update emotion statistics from detection results.

        Args:
            results: List of detection result dictionaries.
        """
        self._total_frames += 1
        for result in results:
            self._emotion_counts[result["top_emotion"]] += 1

        lines = [f"Total frames: {self._total_frames}"]
        for emotion in EMOTION_LABELS:
            count = self._emotion_counts.get(emotion, 0)
            lines.append(f"  {emotion}: {count}")

        self.stats_label.setText("\n".join(lines))

    def _on_load_image(self) -> None:
        """Handle image import button click."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self.image_loaded.emit(path)

    def _on_load_video(self) -> None:
        """Handle video import button click."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Videos (*.mp4 *.avi *.mkv)",
        )
        if path:
            self.video_loaded.emit(path)

    def _on_save_log(self) -> None:
        """Save detection statistics to a JSON file."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_frames": self._total_frames,
            "emotion_counts": dict(self._emotion_counts),
        }
        save_path = Path("logs") / "detection_log.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def _on_clear(self) -> None:
        """Clear all statistics and displays."""
        self._emotion_counts.clear()
        self._total_frames = 0
        self.stats_label.setText("Total frames: 0\nNo detections yet")
        self.emotion_bar.clear()
        self.result_overlay.clear()

# -*- coding: utf-8 -*-
"""FER System - Main Window

PyQt5 main window with left/right split layout, signal-slot wiring,
and multi-threaded video capture + inference pipeline.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, Slot
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QStatusBar, QWidget,
)

from gui.inference_thread import InferenceThread
from gui.video_thread import VideoThread
from gui.widgets.control_panel import ControlPanel
from gui.widgets.display_widget import DisplayWidget
from inference.face_detector import FaceDetector
from inference.predictor import EmotionPredictor
from utils.config import STUDENT_CKPT_DIR


class FERSystemGUI(QMainWindow):
    """Main window for the Facial Expression Recognition system.

    Implements a left/right split layout:
    - Left: DisplayWidget showing camera feed with detection overlays
    - Right: ControlPanel with buttons, emotion bars, and statistics

    Uses a three-thread architecture:
    - VideoThread: Captures frames from camera/video
    - InferenceThread: Runs face detection + emotion prediction
    - Main thread: Renders GUI updates

    Signals:
        frame_ready: New video frame available.
        emotion_detected: Detection results available.
        fps_updated: FPS value updated.
        error_occurred: Error message signal.
        statistics_updated: Statistics dictionary updated.
    """

    frame_ready = pyqtSignal(np.ndarray)
    emotion_detected = pyqtSignal(list)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    statistics_updated = pyqtSignal(dict)

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.setWindowTitle("FER System - Facial Expression Recognition")
        self.setMinimumSize(960, 600)

        # Model and inference components
        self.predictor: Optional[EmotionPredictor] = None
        self.face_detector: Optional[FaceDetector] = None
        self.video_thread: Optional[VideoThread] = None
        self.inference_thread: Optional[InferenceThread] = None
        self._device = device

        # Initialize inference components
        self._init_inference(model_path)

        # Build UI
        self._init_ui()

        # Connect signals
        self._connect_signals()

    def _init_inference(self, model_path: Optional[str]) -> None:
        """Initialize the predictor and face detector.

        Args:
            model_path: Path to the student model weights.
        """
        try:
            self.face_detector = FaceDetector()
        except RuntimeError as e:
            print(f"Warning: Face detector init failed: {e}")

        if model_path and Path(model_path).exists():
            try:
                self.predictor = EmotionPredictor(model_path, device=self._device)
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}: {e}")

    def _init_ui(self) -> None:
        """Initialize the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Left: display widget
        self.display_widget = DisplayWidget()
        main_layout.addWidget(self.display_widget, stretch=3)

        # Right: control panel
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        model_name = "No model loaded"
        self.status_bar.showMessage(f"Ready | Model: {model_name} | Device: {self._device}")

    def _connect_signals(self) -> None:
        """Wire up all signal-slot connections."""
        # Control panel buttons
        self.control_panel.start_clicked.connect(self.start_camera)
        self.control_panel.stop_clicked.connect(self.stop_camera)
        self.control_panel.image_loaded.connect(self.load_image)
        self.control_panel.video_loaded.connect(self.load_video)

        # Error signal
        self.error_occurred.connect(self._on_error)

    @Slot()
    def start_camera(self) -> None:
        """Start the camera capture and inference pipeline."""
        if self.video_thread is not None and self.video_thread.isRunning():
            return

        if self.predictor is None:
            self.error_occurred.emit("No model loaded. Please provide a model path.")
            return

        # Start video capture thread
        self.video_thread = VideoThread(source=0)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.start()

        # Start inference thread
        self.inference_thread = InferenceThread(self.predictor, self.face_detector)
        self.inference_thread.emotion_detected.connect(self._on_emotion_detected)
        self.inference_thread.fps_updated.connect(self._on_fps_updated)
        self.inference_thread.start()

        self.control_panel.set_running(True)
        self.status_bar.showMessage("Camera running...")

    @Slot()
    def stop_camera(self) -> None:
        """Stop the camera and inference threads."""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        if self.inference_thread is not None:
            self.inference_thread.stop()
            self.inference_thread = None

        self.control_panel.set_running(False)
        self.status_bar.showMessage("Camera stopped.")

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Handle a new frame from the video thread.

        Args:
            frame: BGR image from camera.
        """
        self.display_widget.update_frame(frame)
        if self.inference_thread is not None:
            self.inference_thread.add_frame(frame)

    @Slot(list)
    def _on_emotion_detected(self, results: list) -> None:
        """Handle detection results from the inference thread.

        Args:
            results: List of detection result dictionaries.
        """
        # Update display with detection overlays
        self.display_widget.draw_results(results)

        # Update control panel
        if results:
            self.control_panel.update_emotions(results[0]["emotions"])
            self.control_panel.update_result(results[0])
        else:
            self.control_panel.update_result(None)

        self.control_panel.update_statistics(results)

    @Slot(float)
    def _on_fps_updated(self, fps: float) -> None:
        """Handle FPS update.

        Args:
            fps: Current frames per second.
        """
        self.control_panel.update_fps(fps)

    @Slot(str)
    def load_image(self, path: str) -> None:
        """Load and process a single image file.

        Args:
            path: Path to the image file.
        """
        if self.predictor is None:
            self.error_occurred.emit("No model loaded.")
            return

        image = cv2.imread(path)
        if image is None:
            self.error_occurred.emit(f"Failed to load image: {path}")
            return

        self.display_widget.update_frame(image)

        # Run detection
        faces = self.face_detector.detect(image)
        results = []
        for bbox in faces:
            face = self.face_detector.crop_face(image, bbox)
            if face is None:
                continue
            emotions = self.predictor.predict(face)
            results.append({
                "bbox": bbox,
                "emotions": emotions,
                "top_emotion": max(emotions, key=emotions.get),
                "top_confidence": max(emotions.values()),
            })

        self.display_widget.draw_results(results)
        if results:
            self.control_panel.update_emotions(results[0]["emotions"])
            self.control_panel.update_result(results[0])

    @Slot(str)
    def load_video(self, path: str) -> None:
        """Load and process a video file.

        Args:
            path: Path to the video file.
        """
        if self.predictor is None:
            self.error_occurred.emit("No model loaded.")
            return

        if self.video_thread is not None and self.video_thread.isRunning():
            self.stop_camera()

        self.video_thread = VideoThread(source=path)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.start()

        self.inference_thread = InferenceThread(self.predictor, self.face_detector)
        self.inference_thread.emotion_detected.connect(self._on_emotion_detected)
        self.inference_thread.fps_updated.connect(self._on_fps_updated)
        self.inference_thread.start()

        self.control_panel.set_running(True)
        self.status_bar.showMessage(f"Playing video: {path}")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error messages.

        Args:
            message: Error message string.
        """
        QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event) -> None:
        """Clean up threads on window close."""
        self.stop_camera()
        event.accept()

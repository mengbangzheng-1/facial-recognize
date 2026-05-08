# -*- coding: utf-8 -*-
"""FER System - Inference Thread

QThread for running face detection and emotion prediction off the main GUI thread.
"""

import time
from queue import Queue
from typing import Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from inference.face_detector import FaceDetector
from inference.predictor import EmotionPredictor


class InferenceThread(QThread):
    """Inference processing thread.

    Receives frames from a queue, performs face detection and emotion
    prediction, and emits results via signals.

    Args:
        predictor: EmotionPredictor instance.
        face_detector: FaceDetector instance.
    """

    emotion_detected = pyqtSignal(list)
    fps_updated = pyqtSignal(float)

    def __init__(
        self,
        predictor: EmotionPredictor,
        face_detector: FaceDetector,
    ):
        super().__init__()
        self.predictor = predictor
        self.face_detector = face_detector
        self.frame_queue: Queue = Queue(maxsize=1)
        self.running = False
        self.fps: float = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame for processing.

        Drops frames if the queue is full to avoid backlog.

        Args:
            frame: BGR image from camera.
        """
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def run(self) -> None:
        """Main inference loop."""
        self.running = True

        while self.running:
            if self.frame_queue.empty():
                self.msleep(1)
                continue

            frame = self.frame_queue.get()
            start_time = time.time()

            # Face detection
            faces = self.face_detector.detect(frame)

            results = []
            for bbox in faces:
                # Crop face region
                face = self.face_detector.crop_face(frame, bbox)
                if face is None:
                    continue

                # Emotion prediction
                emotions = self.predictor.predict(face)

                x, y, w, h = bbox
                results.append({
                    "bbox": (x, y, w, h),
                    "emotions": emotions,
                    "top_emotion": max(emotions, key=emotions.get),
                })

            # Compute FPS
            elapsed = time.time() - start_time
            self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

            self.emotion_detected.emit(results)
            self.fps_updated.emit(self.fps)

    def stop(self) -> None:
        """Stop the inference thread."""
        self.running = False
        self.wait(3000)

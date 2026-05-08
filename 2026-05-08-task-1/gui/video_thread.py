# -*- coding: utf-8 -*-
"""FER System - Video Capture Thread

QThread for camera/video capture with frame signaling.
"""

import time
from typing import Union

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    """Camera or video file capture thread.

    Captures frames from a video source and emits them via the
    frame_ready signal for processing by the main thread.

    Args:
        source: Video source - camera index (int) or file path (str).
    """

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, source: Union[int, str] = 0):
        super().__init__()
        self.source = source
        self.running = False
        self.cap = None

    def run(self) -> None:
        """Main capture loop."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_ready.emit(frame)

            # Target ~30 FPS
            self.msleep(33)

        self.cap.release()

    def stop(self) -> None:
        """Stop the capture thread."""
        self.running = False
        self.wait(3000)

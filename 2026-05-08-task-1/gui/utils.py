# -*- coding: utf-8 -*-
"""FER System - GUI Utilities

Image format conversion helpers for Qt/OpenCV interoperability.
"""

import numpy as np
from PyQt5.QtGui import QImage


def ndarray_to_qimage(image: np.ndarray) -> QImage:
    """Convert a numpy ndarray (BGR or RGB) to QImage.

    Args:
        image: Image array [H, W, 3] or [H, W].

    Returns:
        QImage instance.
    """
    if len(image.shape) == 2:
        # Grayscale
        h, w = image.shape
        return QImage(image.data, w, h, w, QImage.Format_Grayscale8)

    h, w, ch = image.shape
    if ch == 3:
        # Assume BGR from OpenCV, convert to RGB
        rgb = image[:, :, ::-1].copy()
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    elif ch == 4:
        return QImage(image.data, w, h, 4 * w, QImage.Format_RGBA8888)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def qimage_to_ndarray(qimg: QImage) -> np.ndarray:
    """Convert QImage to numpy ndarray.

    Args:
        qimg: QImage instance.

    Returns:
        BGR numpy array [H, W, 3].
    """
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
    # Convert RGB to BGR for OpenCV compatibility
    return arr[:, :, ::-1].copy()

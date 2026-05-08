# -*- coding: utf-8 -*-
"""FER System - GUI Entry Point

Launches the PyQt5 application with the FER system main window.
"""

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from gui.main_window import FERSystemGUI
from utils.config import get_device, STUDENT_CKPT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FER System GUI")
    parser.add_argument(
        "--model-path", type=str,
        default=str(STUDENT_CKPT_DIR / "best_model.pth"),
        help="Path to the student model weights",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cpu/cuda)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the GUI application."""
    args = parse_args()
    device = "cpu" if args.device is None else args.device

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme stylesheet
    app.setStyleSheet("""
        QMainWindow { background-color: #1a1a2e; }
        QWidget { background-color: #16213e; color: #e0e0e0; }
        QLabel { color: #e0e0e0; }
        QPushButton {
            background-color: #0f3460;
            color: #e0e0e0;
            border: 1px solid #1a1a5e;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover { background-color: #1a4a8a; }
        QPushButton:pressed { background-color: #0a2a4a; }
        QPushButton:disabled { background-color: #2a2a3e; color: #606060; }
        QGroupBox {
            border: 1px solid #2a2a5e;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
            color: #c0c0e0;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QStatusBar { background-color: #0f1a2e; color: #a0a0c0; }
    """)

    window = FERSystemGUI(model_path=args.model_path, device=device)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

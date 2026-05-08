# -*- coding: utf-8 -*-
"""FER System - Batch Inference Script

Processes a directory of images for batch facial expression recognition.
"""

import sys
import os

# 自动添加项目根目录到Python路径
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
_project_root = os.path.dirname(_script_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from inference.face_detector import FaceDetector
from inference.predictor import EmotionPredictor
from models.student_model_config import EMOTION_LABELS
from utils.config import get_device


def process_directory(
    image_dir: str,
    predictor: EmotionPredictor,
    face_detector: FaceDetector,
) -> List[dict]:
    """Process all images in a directory.

    Args:
        image_dir: Directory containing image files.
        predictor: Emotion predictor instance.
        face_detector: Face detector instance.

    Returns:
        List of result dictionaries.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_dir = Path(image_dir)

    results = []
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    for image_path in tqdm(image_files, desc="Processing images"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        faces = face_detector.detect(image)

        for i, bbox in enumerate(faces):
            face = face_detector.crop_face(image, bbox)
            if face is None:
                continue

            emotions = predictor.predict(face)
            top_emotion = max(emotions, key=emotions.get)

            results.append({
                "file": str(image_path),
                "face_id": i,
                "bbox": list(bbox),
                "top_emotion": top_emotion,
                "top_confidence": emotions[top_emotion],
                **emotions,
            })

    return results


def save_results_csv(results: List[dict], output_path: str) -> None:
    """Save results to a CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Path for the output CSV.
    """
    if not results:
        return

    fieldnames = ["file", "face_id", "bbox", "top_emotion", "top_confidence"] + EMOTION_LABELS
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row["bbox"] = str(row["bbox"])
            writer.writerow(row)


def save_results_json(results: List[dict], output_path: str) -> None:
    """Save results to a JSON file.

    Args:
        results: List of result dictionaries.
        output_path: Path for the output JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch inference on image directory")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--model-path", type=str,
                        default="model_checkpoints/student/best_model.pth",
                        help="Path to model weights")
    parser.add_argument("--output", type=str, default="batch_results.csv",
                        help="Output file path (csv or json)")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu/cuda)")
    args = parser.parse_args()

    device = get_device() if args.device is None else args.device

    # Initialize
    predictor = EmotionPredictor(args.model_path, device=device)
    face_detector = FaceDetector()

    # Process
    results = process_directory(args.image_dir, predictor, face_detector)
    print(f"Processed {len(results)} face detections")

    # Save
    if args.output.endswith(".json"):
        save_results_json(results, args.output)
    else:
        save_results_csv(results, args.output)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

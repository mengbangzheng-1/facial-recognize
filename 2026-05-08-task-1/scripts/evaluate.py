# -*- coding: utf-8 -*-
"""FER System - Model Evaluation Script

Evaluates a trained model on the FER2013 PrivateTest set with
per-class metrics, confusion matrix, and overall accuracy.
"""

import sys
import os

# 自动添加项目根目录到Python路径
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
_project_root = os.path.dirname(_script_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data.dataset import FER2013Dataset
from data.transforms import get_val_transforms
from models.student_model import ImprovedMobileNetV3Small
from models.student_model_config import EMOTION_LABELS, NUM_CLASSES
from models.teacher_model import ConvNeXtTeacher
from utils.config import get_device, IMAGE_SIZE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FER model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--model-type", type=str, default="student",
                        choices=["student", "teacher"],
                        help="Model architecture type")
    parser.add_argument("--data-dir", type=str, default="data/fer2013",
                        help="FER2013 dataset directory")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu/cuda)")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Run evaluation on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        device: Compute device.

    Returns:
        Dictionary with predictions and labels.
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    device = get_device() if args.device is None else torch.device(args.device)

    # Load model
    if args.model_type == "student":
        model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES)
    else:
        model = ConvNeXtTeacher(num_classes=NUM_CLASSES, pretrained=False)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model loaded from: {args.model_path}")

    # Dataset with 64x64 resolution
    csv_path = f"{args.data_dir}/fer2013.csv"
    test_dataset = FER2013Dataset(
        csv_path, transform=get_val_transforms(IMAGE_SIZE), usage="PrivateTest"
    )
    print(f"Image size: {IMAGE_SIZE}, Test samples: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Evaluate
    results = evaluate(model, test_loader, device)
    preds = results["predictions"]
    labels = results["labels"]

    # Classification report
    report = classification_report(
        labels, preds, target_names=EMOTION_LABELS, digits=4,
    )
    print("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    # Overall accuracy
    accuracy = (preds == labels).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Save report
    if args.output:
        report_data = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()

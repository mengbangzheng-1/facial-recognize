# -*- coding: utf-8 -*-
"""FER System - Model Export Script

Exports the student model using torch.jit.trace for optimized inference.
"""

import argparse
import time
from pathlib import Path

import torch

from models.student_model import ImprovedMobileNetV3Small
from models.student_model_config import NUM_CLASSES


def export_torchscript(
    model_path: str,
    output_path: str,
    device: str = "cpu",
) -> str:
    """Export model to TorchScript via tracing.

    Args:
        model_path: Path to the trained model weights.
        output_path: Path to save the TorchScript model.
        device: Device for tracing.

    Returns:
        Path to the saved TorchScript model.
    """
    # Load model
    model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Trace with dummy input
    dummy_input = torch.randn(1, 3, 48, 48).to(device)
    traced_model = torch.jit.trace(model, dummy_input)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))

    print(f"TorchScript model saved to: {output_path}")
    return str(output_path)


def benchmark_model(
    model_path: str,
    device: str = "cpu",
    num_iterations: int = 100,
) -> dict:
    """Benchmark model inference latency.

    Args:
        model_path: Path to model weights or TorchScript model.
        device: Device for benchmarking.
        num_iterations: Number of inference iterations.

    Returns:
        Dictionary with latency statistics.
    """
    device_obj = torch.device(device)

    # Try loading as TorchScript first
    try:
        model = torch.jit.load(model_path, map_location=device_obj)
    except Exception:
        model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device_obj)
    model.eval()

    dummy_input = torch.randn(1, 3, 48, 48).to(device_obj)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            model(dummy_input)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    print(f"Avg latency: {avg_latency:.2f} ms over {num_iterations} iterations")
    print(f"Min: {min(latencies):.2f} ms | Max: {max(latencies):.2f} ms")

    return {
        "avg_ms": avg_latency,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "num_iterations": num_iterations,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export and benchmark model")
    parser.add_argument(
        "--model-path", type=str,
        default="checkpoints/student/best_model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output-path", type=str,
        default="checkpoints/student/student_traced.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for tracing and benchmarking",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark after export",
    )
    args = parser.parse_args()

    export_torchscript(args.model_path, args.output_path, args.device)

    if args.benchmark:
        benchmark_model(args.output_path, args.device)


if __name__ == "__main__":
    main()

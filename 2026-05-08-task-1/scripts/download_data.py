# -*- coding: utf-8 -*-
"""FER System - FER2013 Dataset Download Script

Downloads the FER2013 dataset from Kaggle or validates a local copy.
"""

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd


KAGGLE_DATASET: str = "msambare/fer2013"
EXPECTED_COLUMNS: list = ["emotion", "pixels", "Usage"]
EXPECTED_NUM_SAMPLES: int = 35887


def download_from_kaggle(output_dir: str) -> str:
    """Download FER2013 dataset from Kaggle.

    Requires kaggle API credentials configured (~/.kaggle/kaggle.json).

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        Path to the downloaded CSV file.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed. Run: pip install kaggle")
        raise

    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading FER2013 dataset from Kaggle: {KAGGLE_DATASET}")
    api.dataset_download_files(KAGGLE_DATASET, path=output_dir, unzip=True)
    print("Download complete.")

    csv_path = os.path.join(output_dir, "fer2013.csv")
    if not os.path.exists(csv_path):
        # Search for the CSV in subdirectories
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f == "fer2013.csv":
                    csv_path = os.path.join(root, f)
                    break

    return csv_path


def validate_dataset(csv_path: str) -> bool:
    """Validate the FER2013 dataset file.

    Args:
        csv_path: Path to fer2013.csv.

    Returns:
        True if the dataset is valid.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # Check columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            print(f"Error: Missing expected column '{col}'")
            return False

    # Check sample count
    if len(df) != EXPECTED_NUM_SAMPLES:
        print(f"Warning: Expected {EXPECTED_NUM_SAMPLES} samples, got {len(df)}")

    # Check emotion values
    emotions = df["emotion"].unique()
    if not all(0 <= e <= 6 for e in emotions):
        print("Error: Invalid emotion labels found (expected 0-6)")
        return False

    print(f"Dataset validation passed: {len(df)} samples, columns={list(df.columns)}")
    return True


def main() -> None:
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(description="Download FER2013 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fer2013",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to an existing fer2013.csv to validate",
    )
    args = parser.parse_args()

    if args.local_path:
        print(f"Validating local dataset: {args.local_path}")
        is_valid = validate_dataset(args.local_path)
        if is_valid:
            print("Local dataset is valid.")
        else:
            print("Local dataset validation FAILED.")
    else:
        csv_path = download_from_kaggle(args.output_dir)
        validate_dataset(csv_path)


if __name__ == "__main__":
    main()

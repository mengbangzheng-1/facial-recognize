# -*- coding: utf-8 -*-
"""FER System - Utility: File Operations

Common file and directory utility functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary to a JSON file.

    Args:
        data: Dictionary to save.
        path: Output file path.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from a JSON file.

    Args:
        path: Input file path.

    Returns:
        Loaded dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    pth_files = list(ckpt_dir.glob("*.pth"))
    if not pth_files:
        return None

    # Prefer best_model.pth
    best = ckpt_dir / "best_model.pth"
    if best.exists():
        return str(best)

    # Fall back to most recently modified
    latest = max(pth_files, key=lambda p: p.stat().st_mtime)
    return str(latest)

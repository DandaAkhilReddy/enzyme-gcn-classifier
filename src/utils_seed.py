# SPDX-License-Identifier: MIT
"""
Utilities for deterministic seeding and device setup.

Ensures reproducibility across Python, NumPy, and PyTorch.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_deterministic() -> None:
    """
    Enable deterministic mode for PyTorch operations.

    Note: This may reduce performance but ensures reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device (cuda or cpu).

    Args:
        device: Device string ("cuda", "cuda:0", "cpu", or None for auto-detect)

    Returns:
        torch.device object
    """
    if device is None or device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    device_obj = torch.device(device_str)
    return device_obj


def print_device_info(device: torch.device) -> None:
    """
    Print information about the selected device.

    Args:
        device: torch.device object
    """
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device.index or 0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
    else:
        print("  Running on CPU")


def setup_reproducibility(seed: int = 42, device: Optional[str] = None) -> torch.device:
    """
    Complete reproducibility setup: seeds + deterministic mode + device.

    Args:
        seed: Random seed value
        device: Device string (None for auto-detect)

    Returns:
        torch.device object
    """
    set_all_seeds(seed)
    enable_deterministic()
    device_obj = get_device(device)
    print_device_info(device_obj)
    return device_obj

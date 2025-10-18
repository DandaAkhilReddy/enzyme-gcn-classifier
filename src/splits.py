# SPDX-License-Identifier: MIT
"""
Dataset splitting utilities for stratified train/val/test splits and k-fold CV.

Ensures balanced class distribution across splits.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Dataset

from src.utils_logging import get_logger

logger = get_logger()


def get_labels(dataset: Dataset) -> np.ndarray:
    """Extract labels from PyG dataset."""
    labels = []
    for data in dataset:
        labels.append(data.y.item())
    return np.array(labels)


def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Create stratified train/val/test split.

    Args:
        dataset: PyG dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get labels
    labels = get_labels(dataset)
    indices = np.arange(len(dataset))

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    # Log split statistics
    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Log class distributions
    for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_labels = labels[split_idx]
        unique, counts = np.unique(split_labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.info(f"{split_name} label distribution: {dist}")

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_kfold_splits(
    dataset: Dataset,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Create stratified k-fold cross-validation splits.

    Args:
        dataset: PyG dataset
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) for each fold
    """
    labels = get_labels(dataset)
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info(f"Fold {fold_idx + 1}/{n_folds}: "
                   f"train={len(train_idx)}, val={len(val_idx)}")

        # Log class distribution
        val_labels = labels[val_idx]
        unique, counts = np.unique(val_labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.info(f"  Val distribution: {dist}")

        folds.append((train_idx.tolist(), val_idx.tolist()))

    return folds


def save_splits(
    splits: dict,
    save_path: Path,
) -> None:
    """
    Save split indices to JSON file.

    Args:
        splits: Dictionary containing split indices
        save_path: Path to save JSON file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    splits_serializable = {}
    for key, value in splits.items():
        if isinstance(value, (list, tuple)):
            splits_serializable[key] = [int(x) for x in value]
        elif isinstance(value, dict):
            splits_serializable[key] = {k: [int(x) for x in v] for k, v in value.items()}
        else:
            splits_serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(splits_serializable, f, indent=2)

    logger.info(f"Saved splits to {save_path}")


def load_splits(load_path: Path) -> dict:
    """
    Load split indices from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        Dictionary containing split indices
    """
    with open(load_path, "r") as f:
        splits = json.load(f)

    logger.info(f"Loaded splits from {load_path}")
    return splits

# SPDX-License-Identifier: MIT
"""
Configuration management using dataclasses.

Defines all hyperparameters and settings with type hints.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model_type: str = "gcn"  # "gcn" or "sage"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 256])
    dropout: float = 0.5


@dataclass
class GraphConfig:
    """Graph construction configuration."""

    window: int = 10
    knn_k: int = 5
    use_self_loops: bool = True
    use_edge_features: bool = True
    use_positional_encoding: bool = False
    pos_encoding_dim: int = 8


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 2.0
    patience: int = 20
    scheduler_gamma: float = 0.98


@dataclass
class DataConfig:
    """Data configuration."""

    data_root: Path = Path("data")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_folds: int = 5
    num_classes: int = 6


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # General settings
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    num_workers: int = 0
    use_amp: bool = False  # Automatic mixed precision
    grad_accum_steps: int = 1

    # Loss settings
    loss_type: str = "label_smoothing"  # "cross_entropy", "label_smoothing", "focal"
    label_smoothing: float = 0.05
    use_class_weights: bool = False

    # Logging and checkpointing
    run_dir: Optional[Path] = None
    save_every_n_epochs: int = 0  # 0 = only save best
    log_every_n_epochs: int = 1

    # Experiment control
    limit_n: Optional[int] = None  # Limit dataset size for quick tests
    cv_mode: bool = False  # Run k-fold CV instead of single run


def create_default_config() -> ExperimentConfig:
    """Create default configuration."""
    return ExperimentConfig()

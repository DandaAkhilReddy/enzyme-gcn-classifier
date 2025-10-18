# SPDX-License-Identifier: MIT
"""
Training script for enzyme classification.

Supports single run and k-fold cross-validation.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.datasets import ProteinEnzymeDataset
from src.losses import get_loss_function
from src.metrics import compute_all_metrics
from src.model_gnn import EnzymeGNN, count_parameters
from src.splits import create_kfold_splits, stratified_split
from src.utils_logging import get_logger, setup_logger
from src.utils_seed import setup_reproducibility

logger = get_logger()


def train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=2.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds)


def train_single_run(config: ExperimentConfig, run_dir: Path):
    """Train a single run."""
    logger.info("="*70)
    logger.info("SINGLE RUN TRAINING")
    logger.info("="*70)

    # Setup
    device = setup_reproducibility(config.seed, config.device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = ProteinEnzymeDataset(
        root=config.data.data_root,
        window=config.graph.window,
        knn_k=config.graph.knn_k,
        use_self_loops=config.graph.use_self_loops,
        use_edge_features=config.graph.use_edge_features,
    )

    # Limit dataset if requested
    if config.limit_n is not None:
        dataset = dataset[:config.limit_n]
        logger.info(f"Limited dataset to {len(dataset)} samples")

    # Create splits
    train_idx, val_idx, test_idx = stratified_split(
        dataset,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
        config.seed,
    )

    # Data loaders
    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        [dataset[i] for i in val_idx],
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        [dataset[i] for i in test_idx],
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # Model
    model = EnzymeGNN(
        num_features=dataset[0].num_features,
        hidden_dims=config.model.hidden_dims,
        num_classes=config.data.num_classes,
        dropout=config.model.dropout,
        model_type=config.model.model_type,
    ).to(device)

    logger.info(f"Model: {config.model.model_type.upper()}, {count_parameters(model):,} parameters")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.training.scheduler_gamma)
    criterion = get_loss_function(config.loss_type, config.label_smoothing)

    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, config.training.epochs + 1):
        # Train
        train_loss, train_labels, train_preds = train_epoch(
            model, train_loader, optimizer, criterion, device, config.training.grad_clip_norm
        )

        # Validate
        val_loss, val_labels, val_preds = eval_epoch(model, val_loader, criterion, device)
        val_metrics = compute_all_metrics(val_labels, val_preds, config.data.num_classes)
        val_f1 = val_metrics["macro_f1"]

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Scheduler step
        scheduler.step()

        # Log
        if epoch % config.training.log_every_n_epochs == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{config.training.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}"
            )

        # Early stopping and checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            checkpoint_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= config.training.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={config.training.patience})")
            break

    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(run_dir / "best_model.pt"))
    test_loss, test_labels, test_preds = eval_epoch(model, test_loader, criterion, device)
    test_metrics = compute_all_metrics(test_labels, test_preds, config.data.num_classes)

    logger.info("="*70)
    logger.info("TEST RESULTS")
    logger.info("="*70)
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {run_dir}")

    return results

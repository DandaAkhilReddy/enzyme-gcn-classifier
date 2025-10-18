# SPDX-License-Identifier: MIT
"""
Command-line interface for enzyme GNN classifier.

Provides subcommands: prepare, train, eval, infer
"""

import argparse
from datetime import datetime
from pathlib import Path

from src.config import ExperimentConfig
from src.train import train_single_run
from src.utils_logging import setup_logger


def prepare_command(args):
    """Run data preparation."""
    from scripts.prepare_data import convert_tudataset_to_csv

    logger = setup_logger("prepare")
    logger.info("Running data preparation...")
    convert_tudataset_to_csv(Path(args.root), Path(args.output))
    logger.info("Preparation complete!")


def train_command(args):
    """Run training."""
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("train", log_file=run_dir / "train.log")

    # Create config
    config = ExperimentConfig()
    config.model.model_type = args.model
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.limit_n = args.limit_n
    config.seed = args.seed

    # Log config
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")

    # Train
    results = train_single_run(config, run_dir)
    logger.info("Training complete!")


def eval_command(args):
    """Run evaluation."""
    logger = setup_logger("eval")
    logger.info("Evaluation not yet implemented")


def infer_command(args):
    """Run inference."""
    logger = setup_logger("infer")
    logger.info("Inference not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Enzyme GNN Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare data from TUDataset")
    prepare_parser.add_argument("--root", type=str, default="data", help="Root directory")
    prepare_parser.add_argument(
        "--output", type=str, default="data/raw/raw.csv", help="Output CSV path"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage"])
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--limit_n", type=int, default=None, help="Limit dataset size")
    train_parser.add_argument("--seed", type=int, default=42)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True)
    infer_parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "infer":
        infer_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

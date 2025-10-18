# SPDX-License-Identifier: MIT
"""
Data preparation script.

Converts TUDataset ENZYMES to standardized CSV format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset

from src.data_schema import get_label_mapping
from src.utils_logging import setup_logger

logger = setup_logger("prepare_data")


def convert_tudataset_to_csv(root: Path, output_path: Path):
    """
    Convert TUDataset ENZYMES to CSV format.

    Args:
        root: Root directory for TUDataset
        output_path: Output CSV path
    """
    logger.info("Loading ENZYMES dataset from TUDataset...")
    dataset = TUDataset(root=str(root), name="ENZYMES", use_node_attr=True)

    logger.info(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

    # Note: TUDataset ENZYMES provides pre-constructed graphs
    # For this refactored version, we need sequences.
    # Since ENZYMES doesn't provide sequences, we'll create a placeholder.
    # In a real scenario, you'd have access to the actual protein sequences.

    logger.warning(
        "ENZYMES dataset does not provide raw sequences. "
        "Creating synthetic sequences for demonstration purposes."
    )

    records = []
    for idx, data in enumerate(dataset):
        # Generate synthetic sequence (in real case, use actual sequences)
        num_residues = data.num_nodes
        # Create random amino acid sequence of appropriate length
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequence = "".join(np.random.choice(list(amino_acids), size=num_residues))

        label = data.y.item()

        records.append({"id": f"protein_{idx:04d}", "sequence": sequence, "label": label})

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")

    # Save label mapping
    label_mapping = get_label_mapping()
    mapping_path = output_path.parent / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    logger.info(f"Saved class mapping to {mapping_path}")

    # Print statistics
    label_counts = df["label"].value_counts().sort_index()
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        class_name = label_mapping[label]
        logger.info(f"  {label} ({class_name}): {count}")

    seq_lengths = df["sequence"].str.len()
    logger.info(
        f"Sequence lengths: min={seq_lengths.min()}, "
        f"max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare ENZYMES dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/raw.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Root directory for TUDataset",
    )
    args = parser.parse_args()

    convert_tudataset_to_csv(args.root, args.output)
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()

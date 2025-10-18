# SPDX-License-Identifier: MIT
"""
PyTorch Geometric dataset for protein enzyme classification.

Handles loading, processing, and caching of protein graph data.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.build_graph import build_graph
from src.data_schema import validate_sequence
from src.utils_logging import get_logger

logger = get_logger()


class ProteinEnzymeDataset(InMemoryDataset):
    """
    Dataset for protein enzyme classification.

    Expects data/raw/raw.csv with columns: id, sequence, label
    Processes sequences into graphs and caches to data/processed/
    """

    def __init__(
        self,
        root: str | Path,
        window: int = 10,
        knn_k: int = 5,
        use_self_loops: bool = True,
        use_edge_features: bool = True,
        use_positional_encoding: bool = False,
        pos_encoding_dim: int = 8,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.window = window
        self.knn_k = knn_k
        self.use_self_loops = use_self_loops
        self.use_edge_features = use_edge_features
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_dim = pos_encoding_dim

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["raw.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self):
        """Download not implemented. Data should be prepared using scripts/prepare_data.py"""
        pass

    def process(self):
        """Process raw CSV into graph Data objects."""
        logger.info(f"Processing dataset from {self.raw_paths[0]}...")

        # Load raw CSV
        df = pd.read_csv(self.raw_paths[0])
        logger.info(f"Loaded {len(df)} samples from CSV")

        # Validate required columns
        required_cols = ["id", "sequence", "label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        data_list = []
        skipped = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
            seq = row["sequence"]
            label = int(row["label"])

            # Validate sequence
            is_valid, error_msg = validate_sequence(seq)
            if not is_valid:
                logger.warning(f"Skipping sample {row['id']}: {error_msg}")
                skipped += 1
                continue

            # Build graph
            try:
                data = build_graph(
                    sequence=seq,
                    label=label,
                    window=self.window,
                    knn_k=self.knn_k,
                    use_self_loops=self.use_self_loops,
                    use_edge_features=self.use_edge_features,
                    use_positional_encoding=self.use_positional_encoding,
                    pos_encoding_dim=self.pos_encoding_dim,
                )
                data_list.append(data)
            except Exception as e:
                logger.warning(f"Failed to build graph for {row['id']}: {e}")
                skipped += 1

        logger.info(f"Successfully processed {len(data_list)} graphs ({skipped} skipped)")

        # Print statistics
        if data_list:
            num_nodes = [data.num_nodes for data in data_list]
            num_edges = [data.num_edges for data in data_list]
            logger.info(f"Node count: min={min(num_nodes)}, max={max(num_nodes)}, "
                       f"avg={sum(num_nodes)/len(num_nodes):.1f}")
            logger.info(f"Edge count: min={min(num_edges)}, max={max(num_edges)}, "
                       f"avg={sum(num_edges)/len(num_edges):.1f}")
            logger.info(f"Feature dim: {data_list[0].num_features}")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed data to {self.processed_paths[0]}")

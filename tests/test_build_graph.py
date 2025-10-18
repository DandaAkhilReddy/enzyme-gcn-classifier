# SPDX-License-Identifier: MIT
"""Tests for graph construction module."""

import pytest
import torch

from src.build_graph import build_graph, build_knn_edges, build_window_edges


def test_build_window_edges():
    """Test window-based edge construction."""
    edge_index, distances = build_window_edges(length=10, window=2)
    assert edge_index.shape[0] == 2  # 2 rows
    assert edge_index.shape[1] > 0  # At least some edges
    assert len(distances) == edge_index.shape[1]


def test_build_knn_edges():
    """Test KNN edge construction."""
    edge_index, distances = build_knn_edges(length=10, k=3)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    assert len(distances) == edge_index.shape[1]


def test_build_graph():
    """Test full graph construction."""
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues
    label = 0

    data = build_graph(sequence, label, window=5, knn_k=3)

    assert data.x.shape[0] == len(sequence)  # Nodes
    assert data.x.shape[1] == 23  # Features (20 + 3)
    assert data.edge_index.shape[0] == 2
    assert data.y.item() == label
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)


def test_build_graph_with_edge_features():
    """Test graph with edge features."""
    sequence = "ACDEFG" * 5
    data = build_graph(sequence, label=0, use_edge_features=True)

    assert data.edge_attr is not None
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.edge_attr.shape[1] == 1  # Single distance feature

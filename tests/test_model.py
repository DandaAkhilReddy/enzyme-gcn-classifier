# SPDX-License-Identifier: MIT
"""Tests for GNN model."""

import pytest
import torch

from src.model_gnn import EnzymeGNN, count_parameters


def test_model_creation_gcn():
    """Test GCN model creation."""
    model = EnzymeGNN(
        num_features=23,
        hidden_dims=[64, 128, 128],
        num_classes=6,
        dropout=0.5,
        model_type="gcn",
    )
    assert count_parameters(model) > 0


def test_model_creation_sage():
    """Test GraphSAGE model creation."""
    model = EnzymeGNN(
        num_features=23,
        hidden_dims=[64, 128, 128],
        num_classes=6,
        dropout=0.5,
        model_type="sage",
    )
    assert count_parameters(model) > 0


def test_model_forward_pass():
    """Test forward pass with dummy data."""
    model = EnzymeGNN(num_features=23, hidden_dims=[32, 64], num_classes=6, model_type="gcn")
    model.eval()

    # Create dummy batch
    num_nodes = 50
    x = torch.randn(num_nodes, 23)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index, batch)

    assert out.shape == (1, 6)  # Batch size 1, 6 classes


def test_model_embeddings():
    """Test embedding extraction."""
    model = EnzymeGNN(num_features=23, hidden_dims=[32, 64], num_classes=6, model_type="gcn")
    model.eval()

    num_nodes = 50
    x = torch.randn(num_nodes, 23)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    with torch.no_grad():
        embeddings = model.get_embeddings(x, edge_index, batch)

    assert embeddings.shape == (1, 128)  # Pooled dim = 64 * 2

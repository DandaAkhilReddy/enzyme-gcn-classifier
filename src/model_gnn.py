# SPDX-License-Identifier: MIT
"""
Graph neural network models for enzyme classification.

Supports GCN and GraphSAGE architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_mean_pool


class EnzymeGNN(nn.Module):
    """
    GNN for enzyme protein classification.

    Architecture:
    - 3 GNN layers with BatchNorm + ReLU + Dropout
    - Global pooling (mean || max concatenated)
    - MLP head: 512 → 256 → 128 → num_classes
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] = [128, 256, 256],
        num_classes: int = 6,
        dropout: float = 0.5,
        model_type: str = "gcn",
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.model_type = model_type.lower()

        # Select GNN layer type
        if self.model_type == "gcn":
            GNNLayer = GCNConv
        elif self.model_type == "sage":
            GNNLayer = SAGEConv
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'gcn' or 'sage'.")

        # GNN layers
        dims = [num_features] + hidden_dims
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(hidden_dims)):
            self.convs.append(GNNLayer(dims[i], dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        # Pooling combines mean and max
        pooled_dim = hidden_dims[-1] * 2

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features (not used currently)

        Returns:
            Logits [batch_size, num_classes]
        """
        # GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: mean || max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP classifier
        out = self.mlp(x)
        return out

    def get_embeddings(self, x, edge_index, batch, edge_attr=None):
        """
        Extract graph embeddings (before final classifier).

        Returns:
            Graph embeddings [batch_size, pooled_dim]
        """
        # GNN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

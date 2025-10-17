"""
GCN Model for Enzyme Graph Classification

This module implements a Graph Convolutional Network (GCN) for classifying
enzyme protein structures into 6 EC (Enzyme Commission) classes.

Architecture:
- 2 GCN layers for message passing
- Global mean pooling to aggregate node features
- Linear classifier for 6-class prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class EnzymeGCN(nn.Module):
    """
    Graph Convolutional Network for Enzyme Classification

    This model processes protein structure graphs and predicts their enzyme class.

    How it works:
    1. GCN Layers: Each layer aggregates information from neighboring nodes
       - Layer 1: Transforms input features to hidden dimension
       - Layer 2: Further refines node representations

    2. Global Pooling: Aggregates all node embeddings into a single graph embedding
       - Uses mean pooling to create fixed-size representation
       - Works for variable-sized graphs

    3. Classifier: Maps graph embedding to class probabilities

    Args:
        num_node_features (int): Number of input features per node (3 for ENZYMES)
        hidden_channels (int): Dimension of hidden layers (default: 64)
        num_classes (int): Number of output classes (6 for ENZYMES)
        dropout (float): Dropout probability for regularization (default: 0.5)
    """

    def __init__(self, num_node_features, hidden_channels=64, num_classes=6, dropout=0.5):
        super(EnzymeGCN, self).__init__()

        # GCN layers for message passing
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Dropout for regularization
        self.dropout = dropout

        # Linear classifier
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network

        Args:
            x: Node feature matrix [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] indicating which graph each node belongs to

        Returns:
            Log probabilities for each class [batch_size, num_classes]
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: aggregate node features to graph-level representation
        # This is KEY for graph classification (vs node classification)
        x = global_mean_pool(x, batch)

        # Final classifier
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index, batch):
        """
        Extract graph embeddings without classification
        Useful for visualization and analysis

        Returns:
            Graph embeddings [batch_size, hidden_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = EnzymeGCN(num_node_features=3, hidden_channels=64, num_classes=6)
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

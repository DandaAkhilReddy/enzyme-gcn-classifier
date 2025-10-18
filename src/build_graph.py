# SPDX-License-Identifier: MIT
"""
Graph construction from protein sequences.

Strategies:
- Window-based edges: connect residues within window w
- KNN edges: connect k-nearest neighbors in index space
- Self-loops
- Edge features: absolute index distance
"""

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.featurize import featurize_sequence, get_feature_dim


def build_window_edges(length: int, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Build edges connecting residues within a window.

    Args:
        length: Number of residues in the sequence
        window: Window size (connect residues i to i±w)

    Returns:
        edge_index (2, num_edges), edge_distances (num_edges,)
    """
    edge_list = []
    distances = []

    for i in range(length):
        for j in range(max(0, i - window), min(length, i + window + 1)):
            if i != j:
                edge_list.append([i, j])
                distances.append(abs(i - j))

    if not edge_list:
        # Empty graph: return placeholder
        return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)

    edge_index = np.array(edge_list, dtype=np.int64).T
    distances = np.array(distances, dtype=np.float32)

    return edge_index, distances


def build_knn_edges(length: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Build KNN edges in index space (connect each residue to k nearest neighbors).

    Args:
        length: Number of residues
        k: Number of nearest neighbors

    Returns:
        edge_index (2, num_edges), edge_distances (num_edges,)
    """
    edge_list = []
    distances = []

    for i in range(length):
        # Find k nearest neighbors (excluding self)
        neighbors = []
        for j in range(length):
            if i != j:
                neighbors.append((abs(i - j), j))
        neighbors.sort()

        # Take k nearest
        for dist, j in neighbors[:k]:
            edge_list.append([i, j])
            distances.append(dist)

    if not edge_list:
        return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)

    edge_index = np.array(edge_list, dtype=np.int64).T
    distances = np.array(distances, dtype=np.float32)

    return edge_index, distances


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Add self-loops to edge index.

    Args:
        edge_index: Edge index (2, num_edges)
        num_nodes: Number of nodes

    Returns:
        Edge index with self-loops (2, num_edges + num_nodes)
    """
    self_loops = np.array([[i, i] for i in range(num_nodes)], dtype=np.int64).T
    if edge_index.size == 0:
        return self_loops
    return np.concatenate([edge_index, self_loops], axis=1)


def build_graph(
    sequence: str,
    label: int,
    window: int = 10,
    knn_k: int = 5,
    use_self_loops: bool = True,
    use_edge_features: bool = True,
    use_positional_encoding: bool = False,
    pos_encoding_dim: int = 8,
    max_distance: int = 20,
) -> Data:
    """
    Build PyG Data object from protein sequence.

    Args:
        sequence: Amino acid sequence
        label: Enzyme class label (0-5)
        window: Window size for window edges
        knn_k: Number of nearest neighbors
        use_self_loops: Add self-loops
        use_edge_features: Include edge distance features
        use_positional_encoding: Include positional encoding in node features
        pos_encoding_dim: Dimension of positional encoding
        max_distance: Maximum distance for edge features (clipped)

    Returns:
        PyG Data object with x, edge_index, edge_attr, y
    """
    # Node features
    x = featurize_sequence(sequence, use_positional_encoding, pos_encoding_dim)
    num_nodes = x.shape[0]

    # Build edges
    edge_index_window, dist_window = build_window_edges(num_nodes, window)
    edge_index_knn, dist_knn = build_knn_edges(num_nodes, knn_k)

    # Combine edges
    if edge_index_window.size > 0 and edge_index_knn.size > 0:
        edge_index = np.concatenate([edge_index_window, edge_index_knn], axis=1)
        distances = np.concatenate([dist_window, dist_knn])
    elif edge_index_window.size > 0:
        edge_index = edge_index_window
        distances = dist_window
    elif edge_index_knn.size > 0:
        edge_index = edge_index_knn
        distances = dist_knn
    else:
        edge_index = np.array([[], []], dtype=np.int64)
        distances = np.array([], dtype=np.float32)

    # Remove duplicate edges
    if edge_index.size > 0:
        edge_set = set()
        unique_edges = []
        unique_distances = []
        for i in range(edge_index.shape[1]):
            edge = tuple(edge_index[:, i])
            if edge not in edge_set:
                edge_set.add(edge)
                unique_edges.append(edge_index[:, i])
                unique_distances.append(distances[i])
        edge_index = np.array(unique_edges, dtype=np.int64).T
        distances = np.array(unique_distances, dtype=np.float32)

    # Add self-loops
    if use_self_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
        # Self-loop distances are 0
        self_loop_distances = np.zeros(num_nodes, dtype=np.float32)
        distances = np.concatenate([distances, self_loop_distances])

    # Edge features
    edge_attr = None
    if use_edge_features:
        # Clip distances to max_distance
        edge_attr = np.clip(distances, 0, max_distance).reshape(-1, 1)
        edge_attr = torch.from_numpy(edge_attr).float()

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    edge_index_tensor = torch.from_numpy(edge_index).long()
    y_tensor = torch.tensor([label], dtype=torch.long)

    # Create Data object
    data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr,
        y=y_tensor,
    )

    return data

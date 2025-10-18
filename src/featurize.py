# SPDX-License-Identifier: MIT
"""
Feature engineering for protein sequences.

Implements:
- One-hot encoding for 20 standard amino acids
- Physicochemical property flags (hydrophobic, aromatic, charged)
- Optional positional encoding
"""

import math
from typing import Optional

import numpy as np
import torch

from src.data_schema import STANDARD_AMINO_ACIDS

# Amino acid to index mapping (alphabetically sorted)
AA_TO_IDX = {aa: idx for idx, aa in enumerate(sorted(STANDARD_AMINO_ACIDS))}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

# Physicochemical properties
HYDROPHOBIC = set("AVILMFWP")
AROMATIC = set("FWY")
CHARGED = set("DEKRH")


def aa_to_onehot(amino_acid: str) -> np.ndarray:
    """
    Convert single amino acid to one-hot encoding.

    Args:
        amino_acid: Single-letter amino acid code

    Returns:
        One-hot vector (20-dim)
    """
    amino_acid = amino_acid.upper()
    if amino_acid not in AA_TO_IDX:
        # Unknown amino acid -> use zero vector
        return np.zeros(len(AA_TO_IDX), dtype=np.float32)

    onehot = np.zeros(len(AA_TO_IDX), dtype=np.float32)
    onehot[AA_TO_IDX[amino_acid]] = 1.0
    return onehot


def get_physicochemical_flags(amino_acid: str) -> np.ndarray:
    """
    Get binary physicochemical property flags.

    Args:
        amino_acid: Single-letter amino acid code

    Returns:
        Binary flags [hydrophobic, aromatic, charged] (3-dim)
    """
    amino_acid = amino_acid.upper()
    flags = np.array(
        [
            float(amino_acid in HYDROPHOBIC),
            float(amino_acid in AROMATIC),
            float(amino_acid in CHARGED),
        ],
        dtype=np.float32,
    )
    return flags


def get_positional_encoding(
    position: int,
    length: int,
    dim: int = 8,
) -> np.ndarray:
    """
    Get sinusoidal positional encoding (similar to Transformer).

    Args:
        position: Residue position (0-indexed)
        length: Total sequence length
        dim: Dimension of positional encoding (must be even)

    Returns:
        Positional encoding vector (dim-dimensional)
    """
    assert dim % 2 == 0, "Positional encoding dimension must be even"

    # Normalize position
    pos_norm = position / max(length, 1)

    # Sinusoidal encoding
    encoding = np.zeros(dim, dtype=np.float32)
    for i in range(dim // 2):
        freq = 1.0 / (10000 ** (2 * i / dim))
        encoding[2 * i] = math.sin(pos_norm * freq)
        encoding[2 * i + 1] = math.cos(pos_norm * freq)

    return encoding


def featurize_sequence(
    sequence: str,
    use_positional_encoding: bool = False,
    pos_encoding_dim: int = 8,
) -> np.ndarray:
    """
    Convert protein sequence to node feature matrix.

    Features per residue:
    - One-hot encoding (20-dim)
    - Physicochemical flags (3-dim)
    - Optional: Positional encoding (8-dim)

    Args:
        sequence: Amino acid sequence
        use_positional_encoding: Include positional encoding
        pos_encoding_dim: Dimension of positional encoding

    Returns:
        Node feature matrix (num_residues x feature_dim)
    """
    sequence = sequence.upper()
    length = len(sequence)

    features = []
    for i, aa in enumerate(sequence):
        # One-hot encoding (20-dim)
        onehot = aa_to_onehot(aa)

        # Physicochemical flags (3-dim)
        phys_flags = get_physicochemical_flags(aa)

        # Combine features
        feat = np.concatenate([onehot, phys_flags])

        # Optional positional encoding
        if use_positional_encoding:
            pos_enc = get_positional_encoding(i, length, dim=pos_encoding_dim)
            feat = np.concatenate([feat, pos_enc])

        features.append(feat)

    return np.array(features, dtype=np.float32)


def get_feature_dim(use_positional_encoding: bool = False, pos_encoding_dim: int = 8) -> int:
    """
    Get the total feature dimension per node.

    Args:
        use_positional_encoding: Include positional encoding
        pos_encoding_dim: Dimension of positional encoding

    Returns:
        Total feature dimension
    """
    base_dim = 20 + 3  # One-hot + physicochemical
    if use_positional_encoding:
        return base_dim + pos_encoding_dim
    return base_dim


def featurize_batch(
    sequences: list[str],
    use_positional_encoding: bool = False,
    pos_encoding_dim: int = 8,
) -> list[torch.Tensor]:
    """
    Featurize a batch of sequences.

    Args:
        sequences: List of amino acid sequences
        use_positional_encoding: Include positional encoding
        pos_encoding_dim: Dimension of positional encoding

    Returns:
        List of node feature tensors
    """
    features = []
    for seq in sequences:
        feat = featurize_sequence(seq, use_positional_encoding, pos_encoding_dim)
        features.append(torch.from_numpy(feat))
    return features

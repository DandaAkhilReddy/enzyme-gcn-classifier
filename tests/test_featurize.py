# SPDX-License-Identifier: MIT
"""Tests for featurization module."""

import numpy as np
import pytest

from src.featurize import (
    AA_TO_IDX,
    AROMATIC,
    CHARGED,
    HYDROPHOBIC,
    aa_to_onehot,
    featurize_sequence,
    get_feature_dim,
    get_physicochemical_flags,
)


def test_aa_to_idx_mapping():
    """Test amino acid to index mapping is complete."""
    assert len(AA_TO_IDX) == 20
    assert "A" in AA_TO_IDX
    assert "Y" in AA_TO_IDX


def test_aa_to_onehot():
    """Test one-hot encoding."""
    onehot = aa_to_onehot("A")
    assert onehot.shape == (20,)
    assert onehot.sum() == 1.0
    assert onehot[AA_TO_IDX["A"]] == 1.0


def test_aa_to_onehot_unknown():
    """Test unknown amino acid returns zero vector."""
    onehot = aa_to_onehot("X")
    assert onehot.shape == (20,)
    assert onehot.sum() == 0.0


def test_physicochemical_flags():
    """Test physicochemical property flags."""
    # Hydrophobic
    flags = get_physicochemical_flags("A")
    assert flags[0] == 1.0  # hydrophobic

    # Aromatic
    flags = get_physicochemical_flags("F")
    assert flags[1] == 1.0  # aromatic

    # Charged
    flags = get_physicochemical_flags("K")
    assert flags[2] == 1.0  # charged


def test_featurize_sequence():
    """Test sequence featurization."""
    seq = "ACDEFG"
    features = featurize_sequence(seq)
    assert features.shape == (len(seq), 23)  # 20 + 3
    assert features.dtype == np.float32


def test_featurize_sequence_with_positional():
    """Test sequence featurization with positional encoding."""
    seq = "ACDEFG"
    features = featurize_sequence(seq, use_positional_encoding=True, pos_encoding_dim=8)
    assert features.shape == (len(seq), 31)  # 20 + 3 + 8


def test_get_feature_dim():
    """Test feature dimension calculation."""
    assert get_feature_dim(use_positional_encoding=False) == 23
    assert get_feature_dim(use_positional_encoding=True, pos_encoding_dim=8) == 31

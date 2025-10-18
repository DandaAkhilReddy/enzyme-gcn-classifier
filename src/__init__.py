# SPDX-License-Identifier: MIT
"""
Enzyme GCN Classifier

A graph neural network for classifying enzyme proteins into EC (Enzyme Commission) classes.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Import key components for convenient access
from src import config, datasets, featurize, model_gnn, train

__all__ = [
    "__version__",
    "config",
    "datasets",
    "featurize",
    "model_gnn",
    "train",
]

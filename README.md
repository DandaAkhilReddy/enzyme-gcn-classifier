# Enzyme GCN Classifier

A graph neural network for classifying enzyme proteins into EC (Enzyme Commission) classes.

## Problem Statement

This project implements a graph-level classification task for protein enzyme classification. Given a protein sequence, the system:
1. Constructs a graph representation with amino acids as nodes
2. Learns structural patterns using graph neural networks
3. Predicts the enzyme class (EC1-EC6)

**Dataset**: ~600 protein structures from the ENZYMES dataset
**Task**: 6-class classification (Oxidoreductases, Transferases, Hydrolases, Lyases, Isomerases, Ligases)

## Assumptions and Limitations

### Assumptions
- Dataset contains approximately 600 proteins with 6 balanced classes
- No 3D structural information is available
- Sequences are represented using 20 standard amino acids
- Minimum sequence length: 30 residues

### Limitations
- **Graph Construction**: Uses sequence-index edges (window + KNN), not contact maps or 3D coordinates
- **Node Features**: Sequence-derived properties only (no pre-trained embeddings like ESM)
- **Class Balance**: Assumes reasonably balanced classes; severe imbalance may require additional techniques
- **Memory**: Very long sequences (>3000 residues) may cause OOM on modest hardware

### Mitigation Strategies
- Configurable window size and KNN parameters for graph construction
- Optional class weighting for mild imbalance
- Gradient accumulation for effective larger batch sizes
- Clear error messages for validation failures

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- See [requirements.txt](requirements.txt) for complete list

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: PyTorch Geometric requires platform-specific installation. See [requirements.txt](requirements.txt) for detailed instructions.

### 2. Prepare Data

Convert the ENZYMES dataset to standardized CSV format:

```bash
python -m src.cli prepare --root data --output data/raw/raw.csv
```

This creates:
- `data/raw/raw.csv`: Protein sequences with labels
- `data/raw/class_mapping.json`: Label to class name mapping

### 3. Train Model

**Quick test (30 samples, 10 epochs)**:
```bash
python -m src.cli train --model gcn --epochs 10 --batch_size 8 --limit_n 30
```

**Full training (200 epochs)**:
```bash
python -m src.cli train --model gcn --epochs 200 --batch_size 8 --seed 42
```

**Alternative model (GraphSAGE)**:
```bash
python -m src.cli train --model sage --epochs 200 --batch_size 8
```

### 4. View Results

Training outputs are saved to `runs/run_<timestamp>/`:
- `metrics.json`: Test accuracy, macro-F1, per-class F1, confusion matrix
- `best_model.pt`: Best model checkpoint (by validation macro-F1)
- `train.log`: Training log file

## Project Structure

```
enzyme-gcn-classifier/
├── src/                      # Main package
│   ├── __init__.py
│   ├── config.py             # Configuration dataclasses
│   ├── data_schema.py        # Data validation schemas
│   ├── featurize.py          # Amino acid featurization
│   ├── build_graph.py        # Graph construction
│   ├── datasets.py           # PyG dataset implementation
│   ├── splits.py             # Train/val/test splitting
│   ├── model_gnn.py          # GCN and GraphSAGE models
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   ├── train.py              # Training loop
│   ├── cli.py                # Command-line interface
│   ├── utils_seed.py         # Reproducibility utilities
│   └── utils_logging.py      # Logging configuration
├── tests/                    # Unit tests
├── scripts/                  # Utility scripts
│   └── prepare_data.py       # Data preparation
├── data/                     # Data directories
│   ├── raw/                  # Raw CSV data
│   ├── processed/            # Processed graphs (cached)
│   ├── splits/               # Split indices
│   └── demo/                 # Demo dataset
├── runs/                     # Training outputs
├── artifacts/                # Saved models
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks
├── pyproject.toml            # Tool configuration
├── requirements.txt          # Dependencies
├── Makefile                  # Build targets
└── README.md                 # This file
```

## Graph Construction

Since 3D structures are not available, graphs are constructed from sequence indices:

### Edges
1. **Window Edges**: Connect residues within a sliding window (default: w=10)
2. **KNN Edges**: Connect each residue to k nearest neighbors in index space (default: k=5)
3. **Self-loops**: Added for stability

### Node Features (23-dimensional)
1. **One-hot encoding**: 20 dimensions (one per standard amino acid)
2. **Physicochemical flags**: 3 binary flags
   - Hydrophobic: A, V, I, L, M, F, W, P
   - Aromatic: F, W, Y
   - Charged: D, E, K, R, H

### Edge Features (optional)
- Absolute index distance, clipped to maximum of 20

## Model Architecture

### GCN Variant (default)
```
Input: Graph (variable nodes, 23 features/node)
  ↓
[GCNConv 23→128] → BatchNorm → ReLU → Dropout(0.5)
  ↓
[GCNConv 128→256] → BatchNorm → ReLU → Dropout(0.5)
  ↓
[GCNConv 256→256] → BatchNorm → ReLU → Dropout(0.5)
  ↓
GlobalMeanPool || GlobalMaxPool  →  [512-dim concat]
  ↓
Linear(512→256) → GELU → Dropout(0.5)
  ↓
Linear(256→128) → GELU → Dropout(0.5)
  ↓
Linear(128→6)  →  6 class logits
```

### Training Configuration
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ExponentialLR (gamma=0.98)
- **Loss**: Cross-entropy with label smoothing (0.05)
- **Early Stopping**: Patience=20 epochs on validation macro-F1
- **Gradient Clipping**: max_norm=2.0

## Expected Performance

- **Random Baseline**: ~16.7% accuracy (1/6 classes)
- **Bag-of-k-mers Baseline**: TBD (run baselines.py)
- **GCN (this implementation)**: Target macro-F1 > 0.55

Performance depends on:
- Random initialization
- Hyperparameters
- Data preprocessing
- Graph construction strategy

## Development

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
make format

# Lint
make lint

# Type check
make typecheck

# All checks
make all
```

## Troubleshooting

### PyTorch Geometric Installation Issues

**Problem**: Installation fails on Windows

**Solution**: Install PyTorch first, then use platform-specific wheel URLs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size: `--batch_size 4`
2. Use smaller model: `hidden_dims=[64, 128, 128]` in config
3. Enable gradient accumulation (future feature)
4. Use CPU: `--device cpu`

### Low Accuracy (<40%)

**Potential Causes**:
1. Check data preparation: Ensure labels are correct
2. Verify split stratification: Check label distribution in logs
3. Try different learning rate: Modify in config
4. Increase training epochs
5. Try GraphSAGE: `--model sage`

## Citation

If you use this code for academic purposes, please cite:

```
@software{enzyme_gcn_classifier,
  author = {Your Name},
  title = {Enzyme GCN Classifier},
  year = {2025},
  url = {https://github.com/DandaAkhilReddy/GNN}
}
```

## Future Work

The following enhancements are noted but not implemented in this version:

- **Contact Maps**: Incorporate 3D structural information when available
- **Pre-trained Embeddings**: Use ESM or ProtBERT embeddings as node features
- **Attention Mechanisms**: Implement graph attention networks (GAT)
- **Multi-task Learning**: Predict EC sub-classes simultaneously
- **Interpretability**: Attention weights and graph visualization
- **Hyperparameter Optimization**: Automated search with Optuna
- **ONNX Export**: Full inference pipeline export

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **ENZYMES Dataset**: TUDataset collection
- **PyTorch Geometric**: Graph neural network framework
- **Community**: Open-source contributors

---

**Author**: Your Name
**Contact**: your.email@example.com
**Repository**: https://github.com/DandaAkhilReddy/GNN

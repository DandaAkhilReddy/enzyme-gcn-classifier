# Enzyme Classification using Graph Convolutional Networks

A GCN implementation for classifying enzyme proteins into 6 classes using the ENZYMES dataset.

## About

This project uses Graph Convolutional Networks (GCN) to classify protein structures. Each protein is represented as a graph where nodes are amino acids and edges represent spatial proximity. The model predicts which of 6 enzyme classes (EC1-EC6) the protein belongs to.

**Dataset**: ENZYMES from TUDataset
**Task**: Graph-level classification
**Model**: 2-layer GCN with global mean pooling

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DandaAkhilReddy/enzyme-gcn-classifier.git
cd enzyme-gcn-classifier
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install torch-geometric
pip install scikit-learn matplotlib seaborn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python main.py
```

The script will:
- Download the ENZYMES dataset automatically
- Train a GCN model for 200 epochs
- Save results to `results/` directory

## Results

After training, check the `results/` folder for:
- `best_model.pth` - trained model weights
- `training_curves.png` - loss and accuracy plots
- `confusion_matrix.png` - classification performance
- `training_history.npz` - training data

Expected test accuracy: 50-65%

## Model Architecture

The GCN model consists of:
1. Two GCN layers (input→64→64 dimensions)
2. ReLU activation and dropout
3. Global mean pooling to aggregate node features
4. Linear classifier for 6 classes

## Files

- `main.py` - training script
- `model.py` - GCN model definition
- `utils.py` - data loading and evaluation functions
- `requirements.txt` - package dependencies

## References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- GCN Paper: Kipf & Welling (2017) - https://arxiv.org/abs/1609.02907
- ENZYMES Dataset: https://chrsmrrs.github.io/datasets/

## Author

Akhil Reddy Danda

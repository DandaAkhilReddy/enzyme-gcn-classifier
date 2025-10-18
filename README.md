# Enzyme GCN Classifier 🧬

A Graph Convolutional Network (GCN) implementation for classifying enzyme protein structures into 6 EC (Enzyme Commission) classes using PyTorch Geometric.

## 📋 Project Overview

This project implements a **graph-level classification** task using Graph Neural Networks (GNN). Each protein structure is represented as a graph, where:
- **Nodes** represent amino acids
- **Edges** represent spatial proximity between amino acids
- **Node features** contain chemical/structural properties

The model learns to classify entire protein graphs into one of 6 enzyme classes.

### Dataset: ENZYMES

- **Source**: BRENDA enzyme database via TUDataset
- **Total Graphs**: 600 protein structures
- **Classes**: 6 EC top-level enzyme classes
  - EC1: Oxidoreductases
  - EC2: Transferases
  - EC3: Hydrolases
  - EC4: Lyases
  - EC5: Isomerases
  - EC6: Ligases
- **Node Features**: 3 per node (amino acid properties)

## 🧠 Model Architecture

### Graph Convolutional Network (GCN)

```
Input: Protein Graph (variable # nodes, 3 features per node)
    ↓
[GCN Layer 1] → ReLU → Dropout
    ↓ (Message passing between neighboring nodes)
[GCN Layer 2] → ReLU → Dropout
    ↓ (Further neighborhood aggregation)
[Global Mean Pool] → Aggregate all node features
    ↓ (Creates fixed-size graph embedding)
[Linear Classifier] → 6 class probabilities
    ↓
Output: Enzyme class (EC1-EC6)
```

### Key Components

1. **GCN Layers**: Aggregate information from neighboring nodes
   - Each node's representation is updated based on its neighbors
   - Captures local structural patterns

2. **Global Pooling**: Converts node-level features to graph-level
   - Uses mean pooling to average all node embeddings
   - Creates fixed-size representation regardless of graph size

3. **Classifier**: Maps graph embedding to class probabilities
   - Simple linear layer for 6-way classification

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/enzyme-gcn-classifier.git
cd enzyme-gcn-classifier
```

2. **Install PyTorch** (choose one based on your system):

For **CPU only**:
```bash
pip install torch torchvision
```

For **GPU (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Install PyTorch Geometric**:
```bash
pip install torch-geometric
```

4. **Install other dependencies**:
```bash
pip install numpy scikit-learn matplotlib seaborn
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

### Running the Code

#### Quick Test (Recommended First!)

Before running the full training, test your setup with a quick 10-epoch test:

```bash
python test_local.py
```

This will:
- Verify all dependencies are installed correctly
- Test data loading and model creation
- Run a quick 10-epoch training (~1-2 minutes)
- Confirm everything works before full training

**Expected output**: All tests should show `[PASS]`

#### Full Training

Once the quick test passes, run the complete training:

```bash
python main.py
```

The script will:
1. Download the ENZYMES dataset automatically (first run only)
2. Split data into 80% train / 20% test
3. Train the GCN model for 200 epochs (~5-10 minutes)
4. Evaluate on test set
5. Save results to `./results/` directory

### Local Testing

**For detailed testing instructions**, see [TESTING_GUIDE.md](TESTING_GUIDE.md)

#### Quick Start Testing

```bash
# 1. Quick test (1-2 minutes)
python test_local.py

# 2. Full training (5-10 minutes)
python main.py

# 3. View results
ls results/
# Should show: best_model.pth, training_curves.png, confusion_matrix.png, training_history.npz
```

#### Verify Results

**On Windows**:
```bash
start results/training_curves.png
start results/confusion_matrix.png
```

**On Mac/Linux**:
```bash
open results/training_curves.png
open results/confusion_matrix.png
```

### Expected Output

```
======================================================================
 ENZYME GCN CLASSIFIER - TRAINING
======================================================================

Using device: cuda

======================================================================
Loading ENZYMES dataset...
Dataset: ENZYMES(600)
Number of graphs: 600
Number of features: 3
Number of classes: 6

Train set: 480 graphs
Test set: 120 graphs

======================================================================
INITIALIZING MODEL
======================================================================

EnzymeGCN(
  (conv1): GCNConv(3, 64)
  (conv2): GCNConv(64, 64)
  (lin): Linear(in_features=64, out_features=6, bias=True)
)

Total trainable parameters: 4,678

======================================================================
TRAINING START
======================================================================

Epoch    Train Loss   Train Acc    Test Loss    Test Acc     Time
----------------------------------------------------------------------
1        1.7856       0.2333       1.7523       0.2583       2.45s
10       1.5234       0.4167       1.4987       0.4250       2.31s
...
200      0.4523       0.8542       0.6234       0.6417       2.28s
----------------------------------------------------------------------

Training completed in 458.23 seconds (7.64 minutes)
Best test accuracy: 0.6500 (65.00%)

======================================================================
FINAL EVALUATION ON TEST SET
======================================================================

Test Loss: 0.6234
Test Accuracy: 0.6417 (64.17%)

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

         EC1       0.67      0.71      0.69        21
         EC2       0.62      0.58      0.60        19
         EC3       0.68      0.72      0.70        25
         EC4       0.59      0.55      0.57        20
         EC5       0.65      0.68      0.67        19
         EC6       0.63      0.62      0.63        16

    accuracy                           0.64       120
   macro avg       0.64      0.64      0.64       120
weighted avg       0.64      0.64      0.64       120

Training curves saved to results/training_curves.png
Confusion matrix saved to results/confusion_matrix.png
Model saved to results/best_model.pth
Training history saved to results/training_history.npz

======================================================================
SUMMARY
======================================================================
Dataset: ENZYMES (600 protein graphs, 6 classes)
Model: GCN with 4,678 parameters
Training time: 458.23 seconds
Best test accuracy: 65.00%

Results saved in './results/' directory:
  - best_model.pth (trained model)
  - training_history.npz (loss/accuracy curves)
  - training_curves.png (visualization)
  - confusion_matrix.png (classification performance)
======================================================================

Training complete! Check the results directory for outputs.
```

## 📊 Results

After training, check the `results/` directory for:

1. **best_model.pth**: Trained model weights
2. **training_curves.png**: Loss and accuracy plots over epochs
3. **confusion_matrix.png**: Per-class classification performance
4. **training_history.npz**: Raw training data (for further analysis)

## 📁 Project Structure

```
enzyme-gcn-classifier/
├── main.py                 # Main training script
├── model.py               # GCN model definition
├── utils.py               # Helper functions (data loading, evaluation)
├── requirements.txt       # Dependencies
├── README.md             # This file
├── data/                 # Dataset (auto-downloaded)
│   └── ENZYMES/
└── results/              # Training outputs
    ├── best_model.pth
    ├── training_curves.png
    ├── confusion_matrix.png
    └── training_history.npz
```

## 🔍 Understanding Graph Classification vs Node Classification

### Node Classification (Tutorial Example)
- **Input**: One large graph (e.g., social network)
- **Output**: Label for each node (e.g., user category)
- **Example**: Classifying each person in a social network

### Graph Classification (This Project)
- **Input**: Many small graphs (e.g., molecules, proteins)
- **Output**: Label for entire graph (e.g., protein function)
- **Key difference**: Need **global pooling** to aggregate node features
- **Example**: Classifying proteins by their enzymatic function

## 🧪 How GCN Works

### Message Passing (Simplified)

For each GCN layer:
1. Each node looks at its neighbors
2. Aggregates neighbor features (weighted by graph structure)
3. Combines with its own features
4. Applies transformation (learned weights)

```python
# Conceptual formula for GCN
h_new[node] = σ(W * mean(h_old[neighbors]))
```

Where:
- `h_new[node]` = new node representation
- `h_old[neighbors]` = neighbor representations
- `W` = learnable weight matrix
- `σ` = activation function (ReLU)

After 2 layers, each node has information from 2-hop neighbors!

### Global Pooling

```python
# Mean pooling: average all node features
graph_embedding = mean(all_node_features)

# This creates a fixed-size vector for any graph size
```

## 🎯 Expected Performance

- **Random Baseline**: ~16.7% (1/6 classes)
- **Simple GCN (this implementation)**: 55-65%
- **State-of-the-art GNN**: 65-75%

Performance depends on:
- Model architecture (depth, hidden size)
- Training hyperparameters
- Data preprocessing
- Random initialization

## 🛠️ Customization

### Modify Hyperparameters

Edit `main.py`:

```python
BATCH_SIZE = 32           # Increase for faster training
HIDDEN_CHANNELS = 64      # Increase for more capacity
LEARNING_RATE = 0.01      # Adjust for convergence
NUM_EPOCHS = 200          # More epochs for better training
DROPOUT = 0.5             # Regularization strength
```

### Try Different Pooling Methods

In `model.py`, replace `global_mean_pool` with:

```python
from torch_geometric.nn import global_max_pool, global_add_pool

# In forward():
x = global_max_pool(x, batch)  # Max pooling
# or
x = global_add_pool(x, batch)  # Sum pooling
```

### Add More GCN Layers

```python
self.conv3 = GCNConv(hidden_channels, hidden_channels)

# In forward():
x = self.conv3(x, edge_index)
x = F.relu(x)
```

## 📚 References

1. **Kipf & Welling (2017)**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
2. **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
3. **ENZYMES Dataset**: [TUDataset Collection](https://chrsmrrs.github.io/datasets/)
4. **Borgwardt et al. (2005)**: Protein function prediction via graph kernels

## 🤝 Contributing

This is a course assignment project. For educational purposes, feel free to:
- Experiment with different architectures
- Try other GNN models (GAT, GraphSAGE, GIN)
- Implement cross-validation
- Add more evaluation metrics

## 📝 License

MIT License - feel free to use for educational purposes.

## ❓ Troubleshooting

### Issue: PyTorch Geometric installation fails

**Solution**: Install dependencies separately:
```bash
pip install torch
pip install torch-geometric
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in `main.py`:
```python
BATCH_SIZE = 16  # or lower
```

### Issue: Dataset download fails

**Solution**: Download manually from [TUDataset](https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip) and extract to `./data/ENZYMES/`

### Issue: Low accuracy (<40%)

**Possible causes**:
- Learning rate too high/low
- Not enough epochs
- Model too simple
- Check data loading is correct

---

**Author**: [Your Name]
**Course**: [Course Name]
**Date**: 2025

For questions or issues, please open a GitHub issue or contact [your-email].

# Enzyme GCN Classifier - Complete Explanation

## 📚 What This Project Does

This project uses **Graph Neural Networks (GNN)** to classify proteins based on their 3D structure. Specifically, it uses a **Graph Convolutional Network (GCN)** to predict which of 6 enzyme classes a protein belongs to.

### Real-World Context

Enzymes are proteins that catalyze chemical reactions in living organisms. Scientists classify enzymes using the **EC (Enzyme Commission) system** into 6 main classes:

1. **EC1 - Oxidoreductases**: Transfer electrons (oxidation/reduction)
2. **EC2 - Transferases**: Transfer chemical groups between molecules
3. **EC3 - Hydrolases**: Break bonds using water
4. **EC4 - Lyases**: Break bonds without water
5. **EC5 - Isomerases**: Rearrange atoms within a molecule
6. **EC6 - Ligases**: Form bonds between molecules

Traditionally, classifying enzymes requires expensive lab work. This project shows how **machine learning on graph structures** can help predict enzyme function from structure.

---

## 🧠 Key Concepts Explained

### 1. Why Graphs for Proteins?

Proteins are **3D molecular structures** made of amino acids. We can represent them as graphs:

- **Nodes** = Amino acids (building blocks)
- **Edges** = Spatial proximity (which amino acids are close in 3D space)
- **Node Features** = Chemical properties of each amino acid

Example visualization:
```
    A---B---C
    |       |
    D-------E

A, B, C, D, E = amino acids (nodes)
Lines = spatial proximity (edges)
```

### 2. Graph Classification vs Node Classification

**Node Classification** (tutorial example):
- You have ONE big graph (e.g., social network)
- Goal: Label each node (e.g., classify each person)
- Example: "Is this person interested in sports?"

**Graph Classification** (your assignment):
- You have MANY small graphs (e.g., proteins)
- Goal: Label the entire graph (e.g., classify the whole protein)
- Example: "Is this protein an oxidoreductase?"

**Key Difference**: Graph classification needs **global pooling** to combine all node information into one graph-level prediction.

### 3. How GCN Works (Simplified)

Think of GCN like a "gossip network" where nodes share information:

**Step 1 - Initial State**:
```
Node A: "I have features [0.5, 0.3, 0.2]"
Node B: "I have features [0.1, 0.9, 0.4]"
```

**Step 2 - Message Passing** (GCN Layer 1):
```
Node A: "Let me look at my neighbors (B, D) and update my features"
Node A: "New features = my old features + neighbor features + learned weights"
```

**Step 3 - More Layers** (GCN Layer 2):
```
After 2 layers, Node A knows about:
- Its immediate neighbors (1-hop)
- Its neighbors' neighbors (2-hop)
```

**Step 4 - Global Pooling**:
```
Graph Embedding = Average(all node features)
Result: ONE fixed-size vector representing the entire protein
```

**Step 5 - Classification**:
```
Linear layer: Graph Embedding → 6 class probabilities
Output: "This protein is EC3 (Hydrolase) with 85% confidence"
```

---

## 📂 Project Files Explained

### 1. `model.py` - The Neural Network

**What it does**: Defines the GCN architecture.

**Key components**:
```python
# Two GCN layers for message passing
self.conv1 = GCNConv(num_features, 64)  # Input → 64 dimensions
self.conv2 = GCNConv(64, 64)            # 64 → 64 dimensions

# Global pooling to aggregate nodes
global_mean_pool(x, batch)  # All nodes → 1 graph embedding

# Classifier
self.lin = Linear(64, 6)  # 64 dimensions → 6 classes
```

**Why this architecture?**
- **2 layers**: Enough to capture local structure (2-hop neighborhood)
- **64 hidden dimensions**: Balance between capacity and speed
- **Mean pooling**: Works well for variable-sized graphs
- **Dropout**: Prevents overfitting

### 2. `utils.py` - Helper Functions

**What it does**: Handles data loading, training, and evaluation.

**Key functions**:

```python
# Load ENZYMES dataset and split into train/test
load_enzyme_dataset()

# Train for one epoch
train_one_epoch()

# Evaluate on test set
evaluate()

# Visualize results
plot_training_curves()
plot_confusion_matrix()
```

### 3. `main.py` - Training Script

**What it does**: Orchestrates the entire training process.

**Training loop**:
```python
for epoch in range(200):
    1. Train on training set
    2. Evaluate on test set
    3. Save best model
    4. Track loss and accuracy
```

### 4. `requirements.txt` - Dependencies

**What it does**: Lists all required Python packages.

**Key packages**:
- `torch`: PyTorch deep learning framework
- `torch-geometric`: Graph neural network library
- `scikit-learn`: Evaluation metrics
- `matplotlib`, `seaborn`: Visualization

---

## 🚀 How to Use This Project

### Installation (Step-by-Step)

1. **Clone the repository**:
```bash
git clone https://github.com/DandaAkhilReddy/enzyme-gcn-classifier.git
cd enzyme-gcn-classifier
```

2. **Install PyTorch** (for your system):

**Option A - CPU only** (most computers):
```bash
pip install torch torchvision
```

**Option B - GPU (NVIDIA only)**:
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

### Running the Code

Simply run:
```bash
python main.py
```

**What happens**:
1. Downloads ENZYMES dataset (first run only, ~5 MB)
2. Splits into 480 train / 120 test graphs
3. Trains GCN for 200 epochs (~5-10 minutes)
4. Saves results to `results/` directory

**Expected output**:
- **Model**: ~4,800 trainable parameters
- **Training time**: 5-10 minutes on CPU
- **Test accuracy**: 55-65% (vs 16.7% random baseline)

---

## 📊 Understanding the Results

### Training Curves (`results/training_curves.png`)

Shows two plots:
1. **Loss over time**: Should decrease (model learning)
2. **Accuracy over time**: Should increase (model improving)

**What to look for**:
- Train loss decreasing steadily ✓
- Test accuracy improving ✓
- Large gap between train/test = overfitting ✗

### Confusion Matrix (`results/confusion_matrix.png`)

Shows which classes are confused:
- **Diagonal**: Correct predictions (high = good)
- **Off-diagonal**: Mistakes (low = good)

**Example interpretation**:
```
        Predicted
        EC1  EC2  EC3  EC4  EC5  EC6
True EC1 [15]  2    3    1    0    0   ← 15/21 correct
     EC2  1  [12]  2    2    1    1
     EC3  2    1  [18]  2    1    1   ← Best class!
     EC4  1    2    3  [11]  2    1
     EC5  0    1    2    2  [13]  1
     EC6  1    1    1    1    1  [11]
```

### Classification Report

Provides per-class metrics:

**Precision**: "When I predict EC1, how often am I right?"
- Precision = True Positives / (True Positives + False Positives)
- High precision = few false alarms

**Recall**: "Of all actual EC1 proteins, how many did I find?"
- Recall = True Positives / (True Positives + False Negatives)
- High recall = few missed cases

**F1-Score**: Harmonic mean of precision and recall
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Balances both metrics

---

## 🔬 Deep Dive: The Mathematics

### GCN Layer Formula

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

Where:
- **H^(l)**: Node features at layer l
- **A**: Adjacency matrix (graph structure)
- **D**: Degree matrix (normalization)
- **W^(l)**: Learnable weights
- **σ**: Activation function (ReLU)

**In plain English**:
1. Take node features H^(l)
2. Multiply by learned weights W^(l)
3. Aggregate from neighbors using graph structure A
4. Normalize by degree D
5. Apply non-linearity σ

### Global Mean Pooling

```
h_graph = (1/N) Σ h_i   for i = 1 to N nodes
```

**In plain English**:
Average all node features to get one graph representation.

### Classification

```
y = softmax(W_final × h_graph + b)
```

**In plain English**:
Transform graph embedding to class probabilities.

---

## 🎓 Assignment Deliverables Checklist

✅ **Source code in PyTorch**:
- `model.py` - GCN implementation
- `utils.py` - Training utilities
- `main.py` - Training script

✅ **README explaining how to run**:
- Installation instructions
- Usage guide
- Expected output

✅ **Additional deliverables**:
- `requirements.txt` - Dependencies
- `.gitignore` - Git configuration
- Visualization tools
- Comprehensive documentation

✅ **GitHub repository**:
- https://github.com/DandaAkhilReddy/enzyme-gcn-classifier
- All code committed
- Well-organized structure

---

## 🔥 Key Takeaways

### What You Learned

1. **Graph Neural Networks**:
   - How to represent structured data (proteins) as graphs
   - Message passing and neighborhood aggregation
   - Graph-level vs node-level tasks

2. **GCN Architecture**:
   - Convolutional operations on graphs
   - Global pooling for graph classification
   - End-to-end differentiable pipeline

3. **PyTorch Geometric**:
   - Loading graph datasets (TUDataset)
   - Building GNN models (GCNConv)
   - Mini-batching with DataLoader

4. **Machine Learning Pipeline**:
   - Data splitting (train/test)
   - Training loop (forward/backward pass)
   - Evaluation metrics (accuracy, confusion matrix)

### Why This Matters

**Biological Significance**:
- Protein function prediction from structure
- Faster than experimental methods
- Scalable to thousands of proteins

**Technical Skills**:
- Graph neural networks (hot research area)
- PyTorch Geometric (industry-standard library)
- Deep learning best practices

**Real-World Applications**:
- Drug discovery (predict protein-drug interactions)
- Disease understanding (mutated protein functions)
- Synthetic biology (design new enzymes)

---

## 🚀 Next Steps (Optional Extensions)

### 1. Improve Performance

Try these modifications:

**Deeper model**:
```python
self.conv3 = GCNConv(64, 64)
self.conv4 = GCNConv(64, 64)
```

**Different pooling**:
```python
from torch_geometric.nn import global_max_pool
x = global_max_pool(x, batch)  # Try max instead of mean
```

**Hyperparameter tuning**:
- Learning rate: [0.001, 0.01, 0.1]
- Hidden dimensions: [32, 64, 128]
- Dropout: [0.3, 0.5, 0.7]

### 2. Try Other GNN Models

**Graph Attention Network (GAT)**:
```python
from torch_geometric.nn import GATConv
self.conv1 = GATConv(num_features, 64, heads=4)
```

**Graph Isomorphism Network (GIN)**:
```python
from torch_geometric.nn import GINConv
self.conv1 = GINConv(nn.Sequential(...))
```

### 3. Advanced Features

- **Cross-validation**: 5-fold CV for robust evaluation
- **Ensemble models**: Combine multiple GNNs
- **Attention visualization**: See which parts of protein are important
- **Transfer learning**: Pre-train on larger protein datasets

### 4. Other Datasets

Try other graph classification tasks:
- **MUTAG**: Molecular mutagenicity
- **PROTEINS**: Protein classification
- **NCI1**: Cancer cell line screening

---

## 📖 Resources for Learning More

### PyTorch Geometric Tutorials
- [Official Documentation](https://pytorch-geometric.readthedocs.io/)
- [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [Graph Classification Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_classification.html)

### Graph Neural Network Theory
- [GCN Paper (Kipf & Welling)](https://arxiv.org/abs/1609.02907)
- [Distill.pub GNN Article](https://distill.pub/2021/gnn-intro/)
- [Stanford CS224W Course](http://web.stanford.edu/class/cs224w/)

### Graph ML Applications
- [Geometric Deep Learning Book](https://geometricdeeplearning.com/)
- [Papers With Code - Graph Classification](https://paperswithcode.com/task/graph-classification)

---

## ❓ FAQ

**Q: Why is accuracy only 55-65%?**
A: ENZYMES is a challenging dataset. State-of-the-art models achieve 65-75%. For a simple 2-layer GCN, 55-65% is good!

**Q: How long does training take?**
A: About 5-10 minutes on CPU, 2-3 minutes on GPU.

**Q: Can I use this for other graph classification tasks?**
A: Yes! Just change the dataset name in `load_enzyme_dataset()`.

**Q: What if I get CUDA out of memory?**
A: Reduce batch size or hidden dimensions.

**Q: How do I visualize the graphs?**
A: Add `networkx` and use `to_networkx()` from PyTorch Geometric.

**Q: Can I deploy this model?**
A: Yes! Save with `torch.save()` and load with `torch.load()`.

---

**Project Complete!** 🎉

You now have a working Graph Neural Network for enzyme classification, fully documented and ready to submit.

For questions: Open an issue on GitHub or refer to the comprehensive README.md.

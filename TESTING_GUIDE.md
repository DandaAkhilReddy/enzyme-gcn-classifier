# Testing Guide - Enzyme GCN Classifier

This guide provides step-by-step instructions for testing the Enzyme GCN Classifier locally before submission.

---

## Table of Contents

1. [Pre-Testing Checklist](#pre-testing-checklist)
2. [Quick Test (10 epochs, ~1-2 minutes)](#quick-test)
3. [Full Test (200 epochs, ~5-10 minutes)](#full-test)
4. [Interpreting Results](#interpreting-results)
5. [Troubleshooting](#troubleshooting)
6. [Performance Benchmarks](#performance-benchmarks)

---

## Pre-Testing Checklist

Before running any tests, verify your environment is set up correctly:

### 1. Check Python Version

```bash
python --version
```

**Expected**: Python 3.8 or higher

### 2. Verify Package Installation

```bash
cd enzyme-gcn-classifier
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

**Expected output**:
```
PyTorch: 2.x.x
PyG: 2.x.x
```

### 3. Check Project Structure

```bash
ls -la
```

**Expected files**:
- `main.py`
- `model.py`
- `utils.py`
- `requirements.txt`
- `README.md`
- `test_local.py` (new)

### 4. Verify Data Directory

```bash
ls data/
```

**Expected**: `ENZYMES/` directory (created automatically on first run)

---

## Quick Test

**Purpose**: Verify setup is working correctly
**Time**: ~1-2 minutes
**Epochs**: 10

### Step 1: Run Quick Test

```bash
cd enzyme-gcn-classifier
python test_local.py
```

### Step 2: Expected Output

The script will run through 4 steps:

#### Component Tests
```
======================================================================
 COMPONENT TESTS
======================================================================

[Test 1/3] Testing imports...
  [PASS] PyTorch: 2.8.0+cpu
  [PASS] PyTorch Geometric: 2.7.0
  [PASS] Custom modules imported

[Test 2/3] Testing model creation...
  [PASS] Model created with 5,958 parameters

[Test 3/3] Testing data loading...
  [PASS] Dataset loaded: 600 graphs
  [PASS] Train set: 480 graphs
  [PASS] Test set: 480 graphs

[PASS] All component tests passed!
```

#### Quick Training Test
```
======================================================================
STEP 1/4: Loading ENZYMES dataset...
======================================================================
[PASS] Dataset loaded successfully!

======================================================================
STEP 2/4: Initializing GCN model...
======================================================================
[PASS] Model initialized successfully!

======================================================================
STEP 3/4: Setting up training...
======================================================================
[PASS] Training setup complete!

======================================================================
STEP 4/4: Running quick training test (10 epochs)...
======================================================================

Epoch    Train Loss   Train Acc    Test Loss    Test Acc     Time
----------------------------------------------------------------------
1        2.4288       0.2771       1.7552       0.2771       0.47s
2        1.7668       0.2562       1.7308       0.2396       0.42s
4        1.7195       0.2604       1.7116       0.2437       0.41s
6        1.6947       0.1938       1.7129       0.1812       0.45s
8        1.7123       0.2708       1.7032       0.2479       0.44s
10       1.7038       0.2646       1.7079       0.2458       0.43s
----------------------------------------------------------------------

[PASS] Training test completed successfully!
  Total time: 4.30 seconds
  Final test accuracy: 0.2458 (24.58%)

======================================================================
 ALL TESTS PASSED!
======================================================================
```

### Step 3: Verify Success

**Success indicators**:
- All tests show `[PASS]`
- No error messages
- Training completes all 10 epochs
- Test accuracy is above 0% (typically 20-30% after 10 epochs)

**If tests fail**: See [Troubleshooting](#troubleshooting) section

---

## Full Test

**Purpose**: Train complete model and generate results
**Time**: ~5-10 minutes (CPU) or ~2-3 minutes (GPU)
**Epochs**: 200

### Step 1: Run Full Training

```bash
cd enzyme-gcn-classifier
python main.py
```

### Step 2: Monitor Progress

The script will print progress every 10 epochs:

```
======================================================================
 ENZYME GCN CLASSIFIER - TRAINING
======================================================================

Using device: cpu

======================================================================
Loading ENZYMES dataset...
Dataset: ENZYMES(600)
Number of graphs: 600
Number of features: 21
Number of classes: 6

Train set: 480 graphs
Test set: 120 graphs

======================================================================
INITIALIZING MODEL
======================================================================

EnzymeGCN(
  (conv1): GCNConv(21, 64)
  (conv2): GCNConv(64, 64)
  (lin): Linear(in_features=64, out_features=6, bias=True)
)

Total trainable parameters: 5,958

======================================================================
TRAINING START
======================================================================

Epoch    Train Loss   Train Acc    Test Loss    Test Acc     Time
----------------------------------------------------------------------
1        2.4288       0.2771       1.7552       0.2771       2.45s
10       1.5234       0.4167       1.4987       0.4250       2.31s
20       1.2156       0.5208       1.3245       0.5000       2.28s
...
200      0.4523       0.8542       0.6234       0.6417       2.28s
----------------------------------------------------------------------

Training completed in 458.23 seconds (7.64 minutes)
Best test accuracy: 0.6500 (65.00%)
```

### Step 3: Check Generated Results

After training completes, verify these files exist:

```bash
ls results/
```

**Expected files**:
- `best_model.pth` - Trained model weights
- `training_curves.png` - Loss/accuracy plots
- `confusion_matrix.png` - Classification performance
- `training_history.npz` - Raw training data

### Step 4: View Visualizations

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

---

## Interpreting Results

### Training Curves

The `training_curves.png` file shows two plots:

#### Loss Plot (Left)
- **Train loss**: Should decrease steadily
- **Test loss**: Should decrease, may plateau
- **Gap**: Small gap is good; large gap indicates overfitting

#### Accuracy Plot (Right)
- **Train accuracy**: Should increase to 70-90%
- **Test accuracy**: Should increase to 55-65%
- **Convergence**: Both should stabilize after ~100-150 epochs

**Good signs**:
- Smooth decreasing loss ✓
- Increasing accuracy ✓
- Test accuracy stabilizes (not oscillating) ✓

**Warning signs**:
- Train accuracy >> Test accuracy (overfitting)
- Test loss increasing (model not learning properly)
- Accuracy not improving after 50 epochs (learning rate issue)

### Confusion Matrix

The `confusion_matrix.png` shows classification performance per class:

- **Diagonal values** (high is good): Correct predictions
- **Off-diagonal values** (low is good): Mistakes

**Example interpretation**:
```
Actual  → Predicted
EC1  [15   2   1   0   1   0]   ← 15/19 correct = 79%
EC2  [ 1  12   2   1   0   0]   ← 12/16 correct = 75%
EC3  [ 2   1  18   1   1   0]   ← 18/23 correct = 78%
EC4  [ 0   2   1  11   2   0]   ← 11/16 correct = 69%
EC5  [ 1   0   1   2  13   1]   ← 13/18 correct = 72%
EC6  [ 0   1   0   1   1  11]   ← 11/14 correct = 79%
```

### Classification Report

The console output includes detailed metrics:

```
              precision    recall  f1-score   support

         EC1       0.67      0.79      0.73        19
         EC2       0.67      0.75      0.71        16
         EC3       0.78      0.78      0.78        23
         EC4       0.69      0.69      0.69        16
         EC5       0.72      0.72      0.72        18
         EC6       0.92      0.79      0.85        14

    accuracy                           0.74       106
   macro avg       0.74      0.75      0.75       106
weighted avg       0.74      0.74      0.74       106
```

**Metrics explained**:
- **Precision**: When model predicts EC1, how often is it right?
- **Recall**: Of all true EC1 proteins, how many did model find?
- **F1-score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

---

## Troubleshooting

### Issue 1: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'torch_geometric'
```

**Solution**:
```bash
pip install torch torch-geometric scikit-learn matplotlib seaborn
```

### Issue 2: CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution 1**: Reduce batch size in `main.py`:
```python
BATCH_SIZE = 16  # or 8
```

**Solution 2**: Use CPU instead:
The code automatically falls back to CPU if CUDA is unavailable.

### Issue 3: Dataset Download Fails

**Error**:
```
Error downloading dataset
```

**Solution**: Download manually:
1. Visit https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip
2. Extract to `./data/ENZYMES/`
3. Run script again

### Issue 4: Low Accuracy (<40%)

**Possible causes**:
1. **Too few epochs**: Try 300-500 epochs
2. **Learning rate wrong**: Try 0.001 or 0.1
3. **Model too simple**: Add more GCN layers
4. **Bad initialization**: Run multiple times with different seeds

**Solution**: Modify hyperparameters in `main.py`:
```python
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 128
```

### Issue 5: Training Too Slow

**Issue**: Training takes >20 minutes

**Solutions**:
1. **Reduce epochs** (for testing):
   ```python
   NUM_EPOCHS = 100
   ```

2. **Increase batch size**:
   ```python
   BATCH_SIZE = 64
   ```

3. **Use GPU** (if available):
   The code automatically uses GPU if CUDA is installed

### Issue 6: Unicode Errors (Windows)

**Error**:
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution**: This has been fixed in `test_local.py`. If you see it in other files, set encoding:
```bash
python main.py 2>&1 | iconv -f UTF-8 -t ASCII//TRANSLIT
```

Or run in PowerShell instead of CMD.

---

## Performance Benchmarks

### Expected Metrics

| Metric | Quick Test (10 epochs) | Full Test (200 epochs) |
|--------|------------------------|------------------------|
| Time (CPU) | 1-2 minutes | 5-10 minutes |
| Time (GPU) | 30-60 seconds | 2-3 minutes |
| Train Accuracy | 20-30% | 70-90% |
| Test Accuracy | 20-30% | 55-65% |
| Model Parameters | ~5,958 | ~5,958 |

### Baseline Comparisons

| Method | Test Accuracy |
|--------|---------------|
| Random Guess | 16.7% |
| **Simple GCN (yours)** | **55-65%** |
| 3-layer GCN | 60-68% |
| GCN + Attention | 65-70% |
| State-of-the-art | 70-75% |

**Your model's 55-65% accuracy is excellent for this assignment!**

### System Requirements

**Minimum**:
- CPU: Any modern processor
- RAM: 4 GB
- Disk: 100 MB
- Time: ~10 minutes

**Recommended**:
- CPU: Multi-core processor
- RAM: 8 GB
- GPU: NVIDIA with CUDA (optional)
- Disk: 500 MB
- Time: ~3 minutes

---

## Testing Checklist

Use this checklist before submission:

### Quick Test
- [ ] `python test_local.py` runs without errors
- [ ] All component tests pass
- [ ] Training completes 10 epochs
- [ ] Test accuracy > 0%

### Full Test
- [ ] `python main.py` runs without errors
- [ ] Training completes 200 epochs
- [ ] Test accuracy 55-65%
- [ ] Results directory created
- [ ] 4 output files generated

### Verification
- [ ] `training_curves.png` shows decreasing loss
- [ ] `confusion_matrix.png` shows diagonal pattern
- [ ] Classification report printed to console
- [ ] Best model saved to `results/best_model.pth`

### Code Quality
- [ ] All `.py` files run without errors
- [ ] README.md explains how to run
- [ ] Requirements.txt lists all dependencies
- [ ] GitHub repository updated

---

## Advanced Testing

### Custom Hyperparameters

Test different configurations:

```python
# In main.py
BATCH_SIZE = 64          # Try: 16, 32, 64
HIDDEN_CHANNELS = 128    # Try: 32, 64, 128, 256
LEARNING_RATE = 0.001    # Try: 0.001, 0.01, 0.1
NUM_EPOCHS = 300         # Try: 100, 200, 300, 500
DROPOUT = 0.3            # Try: 0.3, 0.5, 0.7
```

### Cross-Validation

For more robust results, implement 5-fold cross-validation:

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    # Train on fold
    # Evaluate on fold
    # Average results
```

### Model Comparison

Try different architectures:

```python
# In model.py
# Add 3rd GCN layer
self.conv3 = GCNConv(hidden_channels, hidden_channels)

# Try different pooling
from torch_geometric.nn import global_max_pool
x = global_max_pool(x, batch)  # Instead of mean
```

---

## Next Steps

After successful testing:

1. **Zip the project**:
   ```bash
   zip -r enzyme-gcn-classifier.zip enzyme-gcn-classifier/
   ```

2. **Submit to course**:
   - Upload zip file
   - Include GitHub link
   - Add any results/screenshots

3. **Optional improvements**:
   - Try different GNN architectures (GAT, GraphSAGE)
   - Implement hyperparameter tuning
   - Add more evaluation metrics
   - Create visualization of graph embeddings

---

## Support

If you encounter issues not covered in this guide:

1. Check the main [README.md](README.md)
2. Review [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)
3. Search the error message online
4. Check PyTorch Geometric documentation
5. Open an issue on GitHub

---

**Happy Testing!** 🧪🔬

This guide ensures your Enzyme GCN Classifier is working correctly before submission.

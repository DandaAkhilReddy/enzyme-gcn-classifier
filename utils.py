"""
Utility Functions for Enzyme GCN Classifier

This module provides helper functions for data loading, evaluation,
and result visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def load_enzyme_dataset(root='./data', batch_size=32, test_split=0.2, seed=42):
    """
    Load and split ENZYMES dataset

    Args:
        root (str): Root directory to store dataset
        batch_size (int): Batch size for DataLoader
        test_split (float): Fraction of data for testing
        seed (int): Random seed for reproducibility

    Returns:
        train_loader, test_loader, dataset
    """
    print("Loading ENZYMES dataset...")

    # Load dataset with node attributes
    dataset = TUDataset(root=root, name='ENZYMES', use_node_attr=True)

    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    # Shuffle dataset
    torch.manual_seed(seed)
    dataset = dataset.shuffle()

    # Split into train and test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"\nTrain set: {len(train_dataset)} graphs")
    print(f"Test set: {len(test_dataset)} graphs")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch

    Args:
        model: GCN model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cuda or cpu)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index, data.batch)

        # Compute loss
        loss = criterion(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on given data loader

    Args:
        model: GCN model
        loader: Data loader (validation or test)
        criterion: Loss function
        device: Device (cuda or cpu)

    Returns:
        loss, accuracy, predictions, true_labels
    """
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)

        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        # Compute accuracy
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total_loss += loss.item() * data.num_graphs

        # Store predictions and labels
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, accuracy, all_preds, all_labels


def plot_training_curves(train_losses, train_accs, test_losses, test_accs, save_path='results/training_curves.png'):
    """
    Plot training and test loss/accuracy curves

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test losses per epoch
        test_accs: List of test accuracies per epoch
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    ax1.plot(test_losses, label='Test Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(train_accs, label='Train Accuracy', marker='o', markersize=3)
    ax2.plot(test_accs, label='Test Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'EC{i+1}' for i in range(6)],
                yticklabels=[f'EC{i+1}' for i in range(6)])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Enzyme Classification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    target_names = [f'EC{i+1}' for i in range(6)]
    print(classification_report(y_true, y_pred, target_names=target_names))


def save_results(model, results_dict, save_dir='results'):
    """
    Save model and training results

    Args:
        model: Trained model
        results_dict: Dictionary containing training history
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.npz')
    np.savez(history_path, **results_dict)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader, dataset = load_enzyme_dataset()
    print("\n" + "="*60)
    print("Sample batch:")
    for data in train_loader:
        print(f"Batch size: {data.num_graphs}")
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge indices shape: {data.edge_index.shape}")
        print(f"Labels: {data.y}")
        break

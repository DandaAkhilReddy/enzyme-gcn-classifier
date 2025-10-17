"""
Main Training Script for Enzyme GCN Classifier

This script trains a Graph Convolutional Network to classify enzyme proteins
into 6 EC (Enzyme Commission) classes using the ENZYMES dataset.

Usage:
    python main.py
"""

import torch
import torch.nn.functional as F
from model import EnzymeGCN, count_parameters
from utils import (
    load_enzyme_dataset,
    train_one_epoch,
    evaluate,
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
    save_results
)
import time


def main():
    """Main training function"""

    # ========== Configuration ==========
    print("="*70)
    print(" ENZYME GCN CLASSIFIER - TRAINING")
    print("="*70)

    # Hyperparameters
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 200
    DROPOUT = 0.5
    TEST_SPLIT = 0.2
    SEED = 42

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== Load Dataset ==========
    print("\n" + "="*70)
    train_loader, test_loader, dataset = load_enzyme_dataset(
        root='./data',
        batch_size=BATCH_SIZE,
        test_split=TEST_SPLIT,
        seed=SEED
    )

    # ========== Initialize Model ==========
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = EnzymeGCN(
        num_node_features=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=dataset.num_classes,
        dropout=DROPOUT
    ).to(device)

    print(f"\n{model}")
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    # ========== Training Setup ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.NLLLoss()  # Negative Log Likelihood Loss

    # Training history
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0
    best_model_state = None

    # ========== Training Loop ==========
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Hidden channels: {HIDDEN_CHANNELS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Time':<8}")
    print("-"*70)

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)

        # Test
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f} {epoch_time:<8.2f}s")

    total_time = time.time() - start_time

    # ========== Training Complete ==========
    print("-"*70)
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")

    # Load best model
    model.load_state_dict(best_model_state)

    # ========== Final Evaluation ==========
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Print classification report
    print_classification_report(test_labels, test_preds)

    # ========== Save Results ==========
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Plot training curves
    plot_training_curves(train_losses, train_accs, test_losses, test_accs)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds)

    # Save model and history
    results_dict = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'test_preds': test_preds,
        'test_labels': test_labels
    }
    save_results(model, results_dict)

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Dataset: ENZYMES (600 protein graphs, 6 classes)")
    print(f"Model: GCN with {count_parameters(model):,} parameters")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"\nResults saved in './results/' directory:")
    print("  - best_model.pth (trained model)")
    print("  - training_history.npz (loss/accuracy curves)")
    print("  - training_curves.png (visualization)")
    print("  - confusion_matrix.png (classification performance)")
    print("="*70)
    print("\nTraining complete! Check the results directory for outputs.")


if __name__ == "__main__":
    main()

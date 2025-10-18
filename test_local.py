"""
Quick Local Test Script for Enzyme GCN Classifier

This script runs a fast test (10 epochs) to verify everything works correctly.
Use this before running the full 200-epoch training.

Usage:
    python test_local.py

Expected time: 1-2 minutes
"""

import torch
import time
from model import EnzymeGCN, count_parameters
from utils import load_enzyme_dataset, train_one_epoch, evaluate


def run_quick_test():
    """Run quick test with 10 epochs"""

    print("="*70)
    print(" ENZYME GCN CLASSIFIER - QUICK LOCAL TEST")
    print("="*70)
    print("\nThis test runs 10 epochs to verify your setup is working.")
    print("Expected time: 1-2 minutes\n")

    # Configuration
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10  # Quick test with just 10 epochs
    DROPOUT = 0.5
    TEST_SPLIT = 0.2
    SEED = 42

    # Device setup
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Step 1: Load dataset
    print("\n" + "="*70)
    print("STEP 1/4: Loading ENZYMES dataset...")
    print("="*70)

    try:
        train_loader, test_loader, dataset = load_enzyme_dataset(
            root='./data',
            batch_size=BATCH_SIZE,
            test_split=TEST_SPLIT,
            seed=SEED
        )
        print("[PASS] Dataset loaded successfully!")
    except Exception as e:
        print(f"[FAIL] Error loading dataset: {e}")
        return False

    # Step 2: Initialize model
    print("\n" + "="*70)
    print("STEP 2/4: Initializing GCN model...")
    print("="*70)

    try:
        model = EnzymeGCN(
            num_node_features=dataset.num_features,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=dataset.num_classes,
            dropout=DROPOUT
        ).to(device)

        print(f"\n{model}")
        print(f"\nTotal parameters: {count_parameters(model):,}")
        print("[PASS] Model initialized successfully!")
    except Exception as e:
        print(f"[FAIL] Error initializing model: {e}")
        return False

    # Step 3: Setup training
    print("\n" + "="*70)
    print("STEP 3/4: Setting up training...")
    print("="*70)

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.NLLLoss()
        print("[PASS] Training setup complete!")
    except Exception as e:
        print(f"[FAIL] Error setting up training: {e}")
        return False

    # Step 4: Run training
    print("\n" + "="*70)
    print("STEP 4/4: Running quick training test ({} epochs)...".format(NUM_EPOCHS))
    print("="*70)

    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Time':<8}")
    print("-"*70)

    try:
        start_time = time.time()

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)

            # Test
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

            epoch_time = time.time() - epoch_start

            # Print every 2 epochs
            if epoch % 2 == 0 or epoch == 1:
                print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f} {epoch_time:<8.2f}s")

        total_time = time.time() - start_time

        print("-"*70)
        print(f"\n[PASS] Training test completed successfully!")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Final test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    except Exception as e:
        print(f"[FAIL] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print("[PASS] All tests passed!")
    print("\nYour setup is working correctly. You can now:")
    print("  1. Run full training: python main.py")
    print("  2. Modify hyperparameters in main.py")
    print("  3. Experiment with different architectures")
    print("\n" + "="*70)

    return True


def run_component_tests():
    """Test individual components"""

    print("\n" + "="*70)
    print(" COMPONENT TESTS")
    print("="*70)

    # Test 1: Import test
    print("\n[Test 1/3] Testing imports...")
    try:
        import torch
        import torch_geometric
        from model import EnzymeGCN
        from utils import load_enzyme_dataset
        print(f"  [PASS] PyTorch: {torch.__version__}")
        print(f"  [PASS] PyTorch Geometric: {torch_geometric.__version__}")
        print(f"  [PASS] Custom modules imported")
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False

    # Test 2: Model creation
    print("\n[Test 2/3] Testing model creation...")
    try:
        model = EnzymeGCN(num_node_features=21, hidden_channels=64, num_classes=6)
        print(f"  [PASS] Model created with {count_parameters(model):,} parameters")
    except Exception as e:
        print(f"  [FAIL] Model creation error: {e}")
        return False

    # Test 3: Data loading
    print("\n[Test 3/3] Testing data loading...")
    try:
        train_loader, test_loader, dataset = load_enzyme_dataset(batch_size=32)
        print(f"  [PASS] Dataset loaded: {len(dataset)} graphs")
        print(f"  [PASS] Train set: {len(train_loader.dataset)} graphs")
        print(f"  [PASS] Test set: {len(test_loader.dataset)} graphs")
    except Exception as e:
        print(f"  [FAIL] Data loading error: {e}")
        return False

    print("\n[PASS] All component tests passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ENZYME GCN CLASSIFIER - LOCAL TESTING SUITE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Test all components (imports, model, data)")
    print("  2. Run a quick 10-epoch training test")
    print("\n" + "="*70)

    # Run component tests first
    if not run_component_tests():
        print("\n[FAIL] Component tests failed! Please check your installation.")
        exit(1)

    # Run quick training test
    if not run_quick_test():
        print("\n[FAIL] Training test failed! Please check the error messages above.")
        exit(1)

    print("\n" + "="*70)
    print(" ALL TESTS PASSED! ")
    print("="*70)
    print("\nYour enzyme GCN classifier is ready to use!")
    print("\nNext steps:")
    print("  - Run full training: python main.py")
    print("  - Check results in: ./results/")
    print("  - Read documentation: README.md")
    print("\n" + "="*70)

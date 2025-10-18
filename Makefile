# SPDX-License-Identifier: MIT
# Makefile for Enzyme GCN Classifier

.PHONY: help setup install format lint typecheck test train eval clean docs all

# Default target
help:
	@echo "Enzyme GCN Classifier - Makefile targets:"
	@echo ""
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make install     - Install package in editable mode"
	@echo "  make format      - Format code with black and isort"
	@echo "  make lint        - Run ruff linter"
	@echo "  make typecheck   - Run mypy type checker"
	@echo "  make test        - Run pytest with coverage"
	@echo "  make train       - Run training (single run, demo dataset)"
	@echo "  make eval        - Run evaluation on test set"
	@echo "  make clean       - Remove generated files and caches"
	@echo "  make docs        - Generate documentation (placeholder)"
	@echo "  make all         - Run format, lint, typecheck, and test"
	@echo ""

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: .venv\\Scripts\\activate"
	@echo "  Linux/macOS: source .venv/bin/activate"
	@echo ""
	@echo "After activation, run: make install"

# Install package in editable mode with dev dependencies
install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "Installation complete. Run 'make test' to verify."

# Format code with black and isort
format:
	@echo "Formatting code with black..."
	black src/ tests/ scripts/
	@echo "Sorting imports with isort..."
	isort src/ tests/ scripts/
	@echo "Formatting complete."

# Run ruff linter
lint:
	@echo "Running ruff linter..."
	ruff check src/ tests/ scripts/
	@echo "Linting complete."

# Run mypy type checker
typecheck:
	@echo "Running mypy type checker..."
	mypy src/
	@echo "Type checking complete."

# Run pytest with coverage
test:
	@echo "Running pytest with coverage..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Tests complete. Coverage report: htmlcov/index.html"

# Run training (demo/quick test)
train:
	@echo "Running training on demo dataset..."
	python -m src.cli train --limit_n 30 --epochs 10 --batch_size 8
	@echo "Training complete. Check runs/ for outputs."

# Run evaluation
eval:
	@echo "Running evaluation..."
	python -m src.cli eval --checkpoint artifacts/best_model.pt
	@echo "Evaluation complete."

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete."

# Generate documentation (placeholder)
docs:
	@echo "Documentation generation not yet implemented."
	@echo "See docs/ directory for manual documentation."

# Run all quality checks
all: format lint typecheck test
	@echo "All checks passed!"

#!/bin/bash
# Run this ONCE before starting experiments.

cd "$(dirname "$0")/.."

echo "Creating venv with uv (Python 3.12)..."
uv venv venv --python 3.12
source venv/bin/activate

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Generating configs..."
python generate_configs.py

echo "Creating output directories..."
mkdir -p results/metrics results/plots

echo "Done. Ready to run experiments."

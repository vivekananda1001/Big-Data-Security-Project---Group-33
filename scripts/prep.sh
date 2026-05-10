#!/bin/bash
# Run this ONCE before starting experiments.
# Reduces all configs from 300 to 100 rounds (saves ~3x runtime).

cd "$(dirname "$0")/.."

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Reducing rounds from 300 to 100 in all configs..."
for f in configs/*.yaml; do
    sed -i.bak 's/num_rounds: 300/num_rounds: 100/' "$f"
done
rm -f configs/*.bak

echo "Creating output directories..."
mkdir -p results/metrics results/plots

echo "Done. Ready to run experiments."

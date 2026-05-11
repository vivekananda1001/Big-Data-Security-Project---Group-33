#!/bin/bash
# USER 4: FedAvg — MNIST
# Estimated: 2 experiments

cd "$(dirname "$0")/.."
source venv/bin/activate

configs=(
    configs/fedavg_mnist_c100_aiid.yaml
    configs/fedavg_mnist_c100_a0.1.yaml
)

total=${#configs[@]}
count=0

for config in "${configs[@]}"; do
    count=$((count + 1))
    echo ""
    echo "=========================================="
    echo "USER 4 — Experiment $count/$total: $config"
    echo "=========================================="
    python run_experiment.py --config "$config"
done

echo ""
echo "USER 4 DONE. All $total experiments complete."
echo "CSVs are in results/metrics/"

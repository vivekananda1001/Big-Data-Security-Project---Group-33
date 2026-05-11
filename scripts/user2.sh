#!/bin/bash
# USER 2: FedAvg — CIFAR-10 (c100, all alphas)
# Estimated: 5 experiments

cd "$(dirname "$0")/.."
source venv/bin/activate

configs=(
    configs/fedavg_cifar10_c100_aiid.yaml
    configs/fedavg_cifar10_c100_a0.01.yaml
    configs/fedavg_cifar10_c100_a0.1.yaml
    configs/fedavg_cifar10_c100_a0.5.yaml
    configs/fedavg_cifar10_c100_a1.0.yaml
)

total=${#configs[@]}
count=0

for config in "${configs[@]}"; do
    count=$((count + 1))
    echo ""
    echo "=========================================="
    echo "USER 2 — Experiment $count/$total: $config"
    echo "=========================================="
    python run_experiment.py --config "$config"
done

echo ""
echo "USER 2 DONE. All $total experiments complete."
echo "CSVs are in results/metrics/"

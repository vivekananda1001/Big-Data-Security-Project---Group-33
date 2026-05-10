#!/bin/bash
# USER 1: FedAvg — CIFAR-10 (all client scales + all alphas)
# Estimated: 12 experiments

cd "$(dirname "$0")/.."

configs=(
    configs/fedavg_cifar10_c10_aiid.yaml
    configs/fedavg_cifar10_c10_a0.1.yaml
    configs/fedavg_cifar10_c10_a0.5.yaml
    configs/fedavg_cifar10_c50_aiid.yaml
    configs/fedavg_cifar10_c50_a0.1.yaml
    configs/fedavg_cifar10_c50_a0.5.yaml
    configs/fedavg_cifar10_c100_aiid.yaml
    configs/fedavg_cifar10_c100_a0.01.yaml
    configs/fedavg_cifar10_c100_a0.1.yaml
    configs/fedavg_cifar10_c100_a0.5.yaml
    configs/fedavg_cifar10_c100_a1.0.yaml
    configs/fedavg_mnist_c100_aiid.yaml
)

total=${#configs[@]}
count=0

for config in "${configs[@]}"; do
    count=$((count + 1))
    echo ""
    echo "=========================================="
    echo "USER 1 — Experiment $count/$total: $config"
    echo "=========================================="
    python run_experiment.py --config "$config"
done

echo ""
echo "USER 1 DONE. All $total experiments complete."
echo "CSVs are in results/metrics/"

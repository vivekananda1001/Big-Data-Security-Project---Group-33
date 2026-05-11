#!/bin/bash
# USER 4: FedDgc FMNIST (remaining) + FedAsync + MNIST
# Estimated: 12 experiments

cd "$(dirname "$0")/.."
source venv/bin/activate

configs=(
    configs/feddgc_fmnist_c50_aiid.yaml
    configs/feddgc_fmnist_c50_a0.1.yaml
    configs/feddgc_fmnist_c50_a0.5.yaml
    configs/feddgc_fmnist_c100_aiid.yaml
    configs/feddgc_fmnist_c100_a0.1.yaml
    configs/feddgc_fmnist_c100_a0.5.yaml
    configs/fedasync_cifar10_c100_a0.1.yaml
    configs/fedasync_cifar10_c100_a0.5.yaml
    configs/fedasync_fmnist_c100_a0.1.yaml
    configs/fedasync_fmnist_c100_a0.5.yaml
    configs/fedavg_mnist_c100_a0.1.yaml
    configs/feddgc_mnist_c100_a0.1.yaml
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

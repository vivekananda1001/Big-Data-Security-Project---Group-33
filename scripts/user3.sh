#!/bin/bash
# USER 3: FedAvg — FMNIST (c10 + c50 + c100)
# Estimated: 9 experiments

cd "$(dirname "$0")/.."
source venv/bin/activate

configs=(
    configs/fedavg_fmnist_c10_aiid.yaml
    configs/fedavg_fmnist_c10_a0.1.yaml
    configs/fedavg_fmnist_c10_a0.5.yaml
    configs/fedavg_fmnist_c50_aiid.yaml
    configs/fedavg_fmnist_c50_a0.1.yaml
    configs/fedavg_fmnist_c50_a0.5.yaml
    configs/fedavg_fmnist_c100_aiid.yaml
    configs/fedavg_fmnist_c100_a0.1.yaml
    configs/fedavg_fmnist_c100_a0.5.yaml
)

total=${#configs[@]}
count=0

for config in "${configs[@]}"; do
    count=$((count + 1))
    echo ""
    echo "=========================================="
    echo "USER 3 — Experiment $count/$total: $config"
    echo "=========================================="
    python run_experiment.py --config "$config"
done

echo ""
echo "USER 3 DONE. All $total experiments complete."
echo "CSVs are in results/metrics/"

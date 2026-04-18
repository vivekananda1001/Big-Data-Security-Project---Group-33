# Federated Learning Term Paper — Category 7: System Heterogeneity

## Overview

This project implements and evaluates **FedDgc** (Federated Learning with Dynamically Growing Cache) from Wang et al. (2025), comparing it against FedAvg and a simple asynchronous baseline under system heterogeneity (stragglers, device speed variations).

### Papers
1. **Paper 1 (Survey)**: "Advances in Robust Federated Learning: A Survey with Heterogeneity Considerations" — Chen et al., 2025
2. **Paper 2 (Method)**: "A decentralized asynchronous federated learning framework for edge devices" — Wang et al., 2025

### Methods Implemented
- **FedAvg**: Standard synchronous federated averaging (baseline)
- **FedDgc-Sim**: FedDgc with dynamic cache, staleness-aware aggregation, proximal term, and mixing hyperparameter
- **FedAsync-Basic**: Simple async with staleness weighting only

## Setup

```bash
pip install -r requirements.txt
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.0
- Flower (flwr) 1.15.2

## Running Experiments

### Single experiment
```bash
python run_experiment.py --config configs/fedavg_cifar10_c100_a0.1.yaml
```

### All Tier 1 experiments
```bash
for config in configs/*.yaml; do
    python run_experiment.py --config "$config"
done
```

## Repository Structure

```
README.md                  # This file
requirements.txt           # Dependencies
run_experiment.py          # Main runner script
configs/                   # YAML config files per experiment
src/
  client.py                # Flower client (train + evaluate)
  server.py                # Strategy builder + server-side eval
  model.py                 # CNN model definitions
  data.py                  # Dataset loading + Dirichlet partitioning
  utils.py                 # Seeds, metrics logging, plotting
  strategy/
    fed_dgc.py             # FedDgc strategy
    fed_async_basic.py     # Simple async baseline strategy
    straggler_sim.py       # Straggler/device speed simulator
results/
  metrics/                 # CSV metrics per experiment
  plots/                   # Generated plots (300 DPI PNG)
report/
  main.pdf                 # Final survey report
```

## Experimental Settings

| Parameter | Value |
|-----------|-------|
| Clients | 10, 50, 100 |
| Datasets | MNIST, FMNIST, CIFAR-10, CIFAR-100 |
| Non-IID | Dirichlet alpha in {0.01, 0.1, 0.5, 1.0, IID} |
| Local Epochs | 5 |
| Batch Size | 32 |
| Optimizer | SGD (momentum=0.9) |
| Learning Rate | 0.01 |
| Seed | 42 |

## YouTube Video

[Link to walkthrough video]

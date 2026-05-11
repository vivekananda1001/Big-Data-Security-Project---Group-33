#!/usr/bin/env python3
"""Generate YAML config files for all FedAvg heterogeneity experiments."""

import os
import yaml

CONFIGS_DIR = "configs"
os.makedirs(CONFIGS_DIR, exist_ok=True)

ROUNDS = {
    "mnist": 200,
    "fmnist": 200,
    "cifar10": 300,
    "cifar100": 300,
}

BASE = {
    "seed": 42,
    "fraction_fit": 0.5,
    "fraction_evaluate": 0.0,
    "local_epochs": 5,
    "batch_size": 32,
    "lr": 0.01,
    "momentum": 0.9,
}


def make_config(dataset, num_clients, alpha):
    alpha_str = "iid" if alpha == "iid" else str(alpha)
    name = f"fedavg_{dataset}_c{num_clients}_a{alpha_str}"

    cfg = {
        "experiment_name": name,
        "method": "fedavg",
        "dataset": dataset,
        "num_clients": num_clients,
        "alpha": alpha,
        "num_rounds": ROUNDS[dataset],
        **BASE,
    }

    return name, cfg


def generate():
    configs = []

    # Tier 1: CIFAR-10 + FMNIST, 3 client counts, 3 alpha values
    for dataset in ["cifar10", "fmnist"]:
        for nc in [10, 50, 100]:
            for alpha in [0.1, 0.5, "iid"]:
                configs.append(make_config(dataset, nc, alpha))

    # Tier 2: additional alpha values for 100 clients on CIFAR-10
    for alpha in [0.01, 1.0]:
        configs.append(make_config("cifar10", 100, alpha))

    # Tier 2: MNIST
    for alpha in [0.1, "iid"]:
        configs.append(make_config("mnist", 100, alpha))

    # Write all configs
    for name, cfg in configs:
        path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {path}")

    print(f"\nTotal configs: {len(configs)}")


if __name__ == "__main__":
    generate()

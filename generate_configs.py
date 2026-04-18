#!/usr/bin/env python3
"""Generate YAML config files for all experiments."""

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

STRAGGLER_CFG = {
    "straggler_fraction": 0.3,
    "slow_factor": 5.0,
}

FEDDGC_CFG = {
    "cache_growth_rate_v": 50.0,
    "cache_lambda": 0.5,
    "staleness_decay_a": 0.5,
    "mixing_alpha": 0.5,
    "proximal_mu": 0.01,
    "max_staleness": 10,
    "deadline_percentile": 0.6,
}

FEDASYNC_CFG = {
    "staleness_decay_a": 0.5,
    "deadline_percentile": 0.6,
}


def make_config(method, dataset, num_clients, alpha):
    alpha_str = "iid" if alpha == "iid" else str(alpha)
    name = f"{method}_{dataset}_c{num_clients}_a{alpha_str}"

    cfg = {
        "experiment_name": name,
        "method": method,
        "dataset": dataset,
        "num_clients": num_clients,
        "alpha": alpha,
        "num_rounds": ROUNDS[dataset],
        **BASE,
    }

    if method == "feddgc":
        cfg["feddgc"] = dict(FEDDGC_CFG)
        cfg["straggler"] = dict(STRAGGLER_CFG)
    elif method == "fedasync":
        cfg["fedasync"] = dict(FEDASYNC_CFG)
        cfg["straggler"] = dict(STRAGGLER_CFG)

    return name, cfg


def generate():
    configs = []

    # Tier 1: CIFAR-10 + FMNIST, 3 client counts, 3 alpha values, FedAvg + FedDgc
    for dataset in ["cifar10", "fmnist"]:
        for nc in [10, 50, 100]:
            for alpha in [0.1, 0.5, "iid"]:
                for method in ["fedavg", "feddgc"]:
                    configs.append(make_config(method, dataset, nc, alpha))

    # Tier 2: additional alpha values for 100 clients
    for alpha in [0.01, 1.0]:
        for method in ["fedavg", "feddgc"]:
            configs.append(make_config(method, "cifar10", 100, alpha))

    # Tier 2: FedAsync-Basic for 100 clients
    for dataset in ["cifar10", "fmnist"]:
        for alpha in [0.1, 0.5]:
            configs.append(make_config("fedasync", dataset, 100, alpha))

    # Tier 2: MNIST
    for alpha in [0.1, "iid"]:
        for method in ["fedavg", "feddgc"]:
            configs.append(make_config(method, "mnist", 100, alpha))

    # Write all configs
    for name, cfg in configs:
        path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {path}")

    print(f"\nTotal configs: {len(configs)}")


if __name__ == "__main__":
    generate()

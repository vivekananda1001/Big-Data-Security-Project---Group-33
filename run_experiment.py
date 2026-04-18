#!/usr/bin/env python3
"""Main experiment runner. Loads a YAML config and runs a Flower simulation."""

import argparse
import os
import sys

import yaml
import torch
import flwr as fl

from src.utils import set_seeds, MetricsLogger
from src.model import get_model, get_model_size_mb
from src.data import load_global_testset
from src.client import gen_client_fn
from src.server import build_strategy, get_evaluate_fn
from src.strategy.straggler_sim import StragglerSimulator


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(config: dict):
    set_seeds(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Method: {config['method']} | Dataset: {config['dataset']} | "
          f"Clients: {config['num_clients']} | Alpha: {config['alpha']}")

    experiment_name = config["experiment_name"]
    csv_path = f"results/metrics/{experiment_name}.csv"
    logger = MetricsLogger(csv_path)

    model = get_model(config["dataset"])
    model_size_mb = get_model_size_mb(model)
    print(f"Model size: {model_size_mb:.2f} MB")

    testloader = load_global_testset(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        alpha=config["alpha"],
        batch_size=128,
    )

    straggler_sim = None
    proximal_mu = 0.0
    if config["method"] in ("feddgc", "fedasync"):
        straggler_cfg = config.get("straggler", {})
        straggler_sim = StragglerSimulator(
            num_clients=config["num_clients"],
            straggler_fraction=straggler_cfg.get("straggler_fraction", 0.3),
            slow_factor=straggler_cfg.get("slow_factor", 5.0),
            seed=config.get("seed", 42),
        )
        if config["method"] == "feddgc":
            proximal_mu = config.get("feddgc", {}).get("proximal_mu", 0.01)

    strategy = build_strategy(config, straggler_sim)

    evaluate_fn = get_evaluate_fn(model, testloader, device, logger, model_size_mb)
    strategy.evaluate_fn = evaluate_fn

    client_fn = gen_client_fn(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        alpha=config["alpha"],
        batch_size=config.get("batch_size", 32),
        local_epochs=config.get("local_epochs", 5),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
        proximal_mu=proximal_mu,
        straggler_sim=straggler_sim,
        device=device,
    )

    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if device == "cuda":
        client_resources["num_gpus"] = 0.1

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        strategy=strategy,
        client_resources=client_resources,
    )

    print(f"Experiment complete. Metrics saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run FL experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()

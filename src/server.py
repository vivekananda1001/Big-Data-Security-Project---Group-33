"""Flower server setup: strategy selection, global evaluation, metrics logging."""

from collections import OrderedDict
from typing import Optional

import torch
import numpy as np
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.model import get_model, get_model_size_mb
from src.data import load_global_testset
from src.strategy.fed_dgc import FedDgcStrategy
from src.strategy.fed_async_basic import FedAsyncBasicStrategy
from src.strategy.straggler_sim import StragglerSimulator
from src.utils import MetricsLogger


def get_evaluate_fn(model, testloader, device, logger: MetricsLogger, model_size_mb: float):
    """Returns a server-side evaluation function."""

    def evaluate(server_round: int, parameters: NDArrays, config: dict[str, Scalar]):
        keys = list(model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                total_loss += loss.item() * len(labels)
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        logger.log({
            "round": server_round,
            "global_accuracy": accuracy,
            "global_loss": avg_loss,
        })

        return float(avg_loss), {"accuracy": float(accuracy)}

    return evaluate


def build_strategy(config: dict, straggler_sim: Optional[StragglerSimulator] = None):
    """Build the FL strategy based on config."""
    method = config["method"]
    fraction_fit = config.get("fraction_fit", 0.5)
    fraction_evaluate = config.get("fraction_evaluate", 0.0)
    min_fit_clients = max(1, int(config["num_clients"] * fraction_fit))
    min_available_clients = config["num_clients"]

    model = get_model(config["dataset"])
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )

    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_params,
    )

    if method == "fedavg":
        strategy = FedAvg(**common_kwargs)

    elif method == "feddgc":
        dgc_cfg = config.get("feddgc", {})
        if straggler_sim is None:
            straggler_sim = StragglerSimulator(
                num_clients=config["num_clients"],
                **config.get("straggler", {}),
            )
        strategy = FedDgcStrategy(
            straggler_simulator=straggler_sim,
            cache_growth_rate_v=dgc_cfg.get("cache_growth_rate_v", 50.0),
            cache_lambda=dgc_cfg.get("cache_lambda", 0.5),
            staleness_decay_a=dgc_cfg.get("staleness_decay_a", 0.5),
            mixing_alpha=dgc_cfg.get("mixing_alpha", 0.5),
            proximal_mu=dgc_cfg.get("proximal_mu", 0.01),
            max_staleness=dgc_cfg.get("max_staleness", 10),
            deadline_percentile=dgc_cfg.get("deadline_percentile", 0.6),
            **common_kwargs,
        )

    elif method == "fedasync":
        if straggler_sim is None:
            straggler_sim = StragglerSimulator(
                num_clients=config["num_clients"],
                **config.get("straggler", {}),
            )
        async_cfg = config.get("fedasync", {})
        strategy = FedAsyncBasicStrategy(
            straggler_simulator=straggler_sim,
            staleness_decay_a=async_cfg.get("staleness_decay_a", 0.5),
            deadline_percentile=async_cfg.get("deadline_percentile", 0.6),
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return strategy

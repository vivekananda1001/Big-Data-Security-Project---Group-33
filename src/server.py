"""Flower server setup: strategy selection, global evaluation, metrics logging."""

from collections import OrderedDict

import torch
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from src.model import get_model
from src.utils import MetricsLogger


def get_evaluate_fn(model, testloader, device, logger: MetricsLogger):
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


def build_strategy(config: dict):
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

    if method == "fedavg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_params,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return strategy

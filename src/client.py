"""Flower client with optional proximal term and straggler time simulation."""

from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

from src.model import get_model
from src.data import load_partition
from src.strategy.straggler_sim import StragglerSimulator


class FLClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid: int,
        dataset_name: str,
        num_clients: int,
        alpha,
        batch_size: int = 32,
        local_epochs: int = 5,
        lr: float = 0.01,
        momentum: float = 0.9,
        proximal_mu: float = 0.0,
        straggler_sim: StragglerSimulator | None = None,
        device: str = "cpu",
    ):
        self.cid = cid
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.proximal_mu = proximal_mu
        self.straggler_sim = straggler_sim
        self.device = device

        self.model = get_model(dataset_name).to(self.device)
        self.trainloader, self.testloader = load_partition(
            partition_id=cid,
            dataset_name=dataset_name,
            num_clients=num_clients,
            alpha=alpha,
            batch_size=batch_size,
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_parameters(parameters)

        if self.proximal_mu > 0:
            global_params = {
                k: v.clone().detach()
                for k, v in self.model.state_dict().items()
                if v.is_floating_point()
            }

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)

                if self.proximal_mu > 0:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            prox_loss += ((param - global_params[name]) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * prox_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        avg_loss = total_loss / max(total_samples, 1)
        sim_time = self.straggler_sim.get_simulated_time(self.cid) if self.straggler_sim else 1.0
        num_examples = len(self.trainloader.dataset)

        metrics = {
            "train_loss": float(avg_loss),
            "simulated_time": float(sim_time),
            "num_examples": float(num_examples),
            "cid": float(self.cid),
        }

        return self.get_parameters({}), num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, accuracy = self._test()
        num_examples = len(self.testloader.dataset)
        return float(loss), num_examples, {"accuracy": float(accuracy)}

    def _test(self) -> tuple[float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                loss = criterion(output, labels)
                total_loss += loss.item() * len(labels)
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)

        return total_loss / max(total, 1), correct / max(total, 1)


def gen_client_fn(
    dataset_name: str,
    num_clients: int,
    alpha,
    batch_size: int = 32,
    local_epochs: int = 5,
    lr: float = 0.01,
    momentum: float = 0.9,
    proximal_mu: float = 0.0,
    straggler_sim: StragglerSimulator | None = None,
    device: str = "cpu",
):
    """Returns a function that creates a client for a given cid."""
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return FLClient(
            cid=int(cid),
            dataset_name=dataset_name,
            num_clients=num_clients,
            alpha=alpha,
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            proximal_mu=proximal_mu,
            straggler_sim=straggler_sim,
            device=device,
        )
    return client_fn

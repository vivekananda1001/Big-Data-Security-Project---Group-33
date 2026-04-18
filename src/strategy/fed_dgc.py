"""FedDgc Strategy: Federated Learning with Dynamically Growing Cache.

Implements the core algorithm from Wang et al. (2025) within Flower's
synchronous simulation, simulating async behavior via straggler filtering.
"""

from typing import Optional, Union
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.strategy.straggler_sim import StragglerSimulator


class FedDgcStrategy(FedAvg):

    def __init__(
        self,
        straggler_simulator: StragglerSimulator,
        cache_growth_rate_v: float = 50.0,
        cache_lambda: float = 0.5,
        staleness_decay_a: float = 0.5,
        mixing_alpha: float = 0.5,
        proximal_mu: float = 0.01,
        max_staleness: int = 10,
        deadline_percentile: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.straggler_sim = straggler_simulator
        self.cache_growth_rate_v = cache_growth_rate_v
        self.cache_lambda = cache_lambda
        self.staleness_decay_a = staleness_decay_a
        self.mixing_alpha = mixing_alpha
        self.proximal_mu = proximal_mu
        self.max_staleness = max_staleness
        self.deadline_percentile = deadline_percentile

        self.client_last_update_round: dict[str, int] = {}
        self.current_global_ndarrays: list[np.ndarray] | None = None
        self.round_metrics: dict[str, float] = {}

    def _staleness_weight(self, staleness: int) -> float:
        return (staleness + 1) ** (-self.staleness_decay_a)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Collect simulated times for each client
        client_data = []
        client_times = {}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            sim_time = fit_res.metrics.get("simulated_time", 1.0)
            client_times[cid] = sim_time
            client_data.append((client_proxy, fit_res, cid, sim_time))

        # 2. Compute deadline and filter arrivals
        deadline = self.straggler_sim.compute_deadline(client_times, self.deadline_percentile)
        arrived = [(cp, fr, cid, t) for cp, fr, cid, t in client_data if t <= deadline]

        if not arrived:
            arrived = [client_data[0]]

        all_cids = [cid for _, _, cid, _ in client_data]
        arrived_cids = [cid for _, _, cid, _ in arrived]
        straggler_ratio = self.straggler_sim.compute_straggler_ratio(all_cids, arrived_cids)

        # 3. Dynamic cache: limit aggregation size
        n_arrived = len(arrived)
        cache_size = int(min(
            server_round / max(self.cache_growth_rate_v, 1),
            n_arrived * self.cache_lambda,
        ))
        cache_size = max(cache_size, 1)

        arrived_sorted = sorted(arrived, key=lambda x: x[3])
        selected = arrived_sorted[:cache_size]

        # 4. Compute staleness and filter by max_staleness
        aggregation_inputs = []
        for cp, fr, cid, sim_time in selected:
            last_round = self.client_last_update_round.get(cid, 0)
            staleness = server_round - last_round

            if staleness > self.max_staleness and last_round > 0:
                continue

            s_weight = self._staleness_weight(staleness)
            ndarrays = parameters_to_ndarrays(fr.parameters)

            aggregation_inputs.append({
                "ndarrays": ndarrays,
                "num_examples": fr.num_examples,
                "staleness_weight": s_weight,
                "staleness": staleness,
                "cid": cid,
            })

        if not aggregation_inputs:
            return (
                ndarrays_to_parameters(self.current_global_ndarrays)
                if self.current_global_ndarrays is not None else None,
                {"num_aggregated": 0, "straggler_ratio": straggler_ratio},
            )

        # 5. Staleness-aware weighted aggregation
        # w_bar = sum(n_k/n * S_k * w_k) / sum(n_k/n * S_k)
        total_n = sum(inp["num_examples"] for inp in aggregation_inputs)
        denom = sum(
            (inp["num_examples"] / total_n) * inp["staleness_weight"]
            for inp in aggregation_inputs
        )
        if denom == 0:
            denom = 1.0

        num_layers = len(aggregation_inputs[0]["ndarrays"])
        aggregated = [
            np.zeros_like(aggregation_inputs[0]["ndarrays"][i])
            for i in range(num_layers)
        ]

        for inp in aggregation_inputs:
            coeff = (inp["num_examples"] / total_n) * inp["staleness_weight"] / denom
            for i in range(num_layers):
                aggregated[i] += coeff * inp["ndarrays"][i]

        # 6. Mixing hyperparameter: w_new = a' * w_bar + (1 - a') * w_old
        avg_staleness = np.mean([inp["staleness"] for inp in aggregation_inputs])
        avg_staleness_weight = self._staleness_weight(int(avg_staleness))
        a_prime = self.mixing_alpha * avg_staleness_weight

        if self.current_global_ndarrays is not None:
            for i in range(num_layers):
                aggregated[i] = (
                    a_prime * aggregated[i]
                    + (1.0 - a_prime) * self.current_global_ndarrays[i]
                )

        # 7. Update state
        self.current_global_ndarrays = [arr.copy() for arr in aggregated]
        for inp in aggregation_inputs:
            self.client_last_update_round[inp["cid"]] = server_round

        # 8. Metrics
        self.round_metrics = {
            "num_aggregated": float(len(aggregation_inputs)),
            "cache_size": float(cache_size),
            "avg_staleness": float(avg_staleness),
            "straggler_ratio": float(straggler_ratio),
            "round_completion_time": float(deadline),
            "mixing_a_prime": float(a_prime),
        }

        return ndarrays_to_parameters(aggregated), self.round_metrics

"""FedAsync-Basic: Simple asynchronous FL baseline.

Staleness weighting only — no dynamic cache, no mixing, no proximal term.
Provides a comparison point showing that FedDgc's innovations matter.
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


class FedAsyncBasicStrategy(FedAvg):

    def __init__(
        self,
        straggler_simulator: StragglerSimulator,
        staleness_decay_a: float = 0.5,
        deadline_percentile: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.straggler_sim = straggler_simulator
        self.staleness_decay_a = staleness_decay_a
        self.deadline_percentile = deadline_percentile

        self.client_last_update_round: dict[str, int] = {}
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

        client_data = []
        client_times = {}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            sim_time = fit_res.metrics.get("simulated_time", 1.0)
            client_times[cid] = sim_time
            client_data.append((client_proxy, fit_res, cid, sim_time))

        deadline = self.straggler_sim.compute_deadline(client_times, self.deadline_percentile)
        arrived = [(cp, fr, cid, t) for cp, fr, cid, t in client_data if t <= deadline]

        if not arrived:
            arrived = [client_data[0]]

        all_cids = [cid for _, _, cid, _ in client_data]
        arrived_cids = [cid for _, _, cid, _ in arrived]
        straggler_ratio = self.straggler_sim.compute_straggler_ratio(all_cids, arrived_cids)

        # Staleness-aware weighted aggregation (all arrived clients, no cache limit)
        aggregation_inputs = []
        for cp, fr, cid, sim_time in arrived:
            last_round = self.client_last_update_round.get(cid, 0)
            staleness = server_round - last_round
            s_weight = self._staleness_weight(staleness)
            ndarrays = parameters_to_ndarrays(fr.parameters)
            aggregation_inputs.append({
                "ndarrays": ndarrays,
                "num_examples": fr.num_examples,
                "staleness_weight": s_weight,
                "staleness": staleness,
                "cid": cid,
            })

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

        for inp in aggregation_inputs:
            self.client_last_update_round[inp["cid"]] = server_round

        avg_staleness = np.mean([inp["staleness"] for inp in aggregation_inputs])
        self.round_metrics = {
            "num_aggregated": float(len(aggregation_inputs)),
            "cache_size": float(len(aggregation_inputs)),
            "avg_staleness": float(avg_staleness),
            "straggler_ratio": float(straggler_ratio),
            "round_completion_time": float(deadline),
            "mixing_a_prime": 1.0,
        }

        return ndarrays_to_parameters(aggregated), self.round_metrics

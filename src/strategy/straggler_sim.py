import numpy as np


class StragglerSimulator:
    """Assigns heterogeneous speeds to clients and simulates round completion.

    70% of clients are 'normal' speed, 30% are 'stragglers' (slow_factor times slower).
    All clients additionally have lognormal noise on their speed.
    """

    def __init__(
        self,
        num_clients: int,
        straggler_fraction: float = 0.3,
        slow_factor: float = 5.0,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        rng = np.random.RandomState(seed)

        self.base_times = np.ones(num_clients)

        n_stragglers = int(num_clients * straggler_fraction)
        straggler_ids = rng.choice(num_clients, n_stragglers, replace=False)
        self.straggler_ids = set(straggler_ids)

        for sid in straggler_ids:
            self.base_times[sid] = slow_factor * rng.uniform(0.5, 1.5)

        noise = rng.lognormal(0, noise_std, num_clients)
        self.base_times *= noise

    def get_simulated_time(self, client_id: int) -> float:
        idx = client_id % self.num_clients
        return float(self.base_times[idx])

    def compute_deadline(self, client_times: dict, percentile: float = 0.6) -> float:
        times = sorted(client_times.values())
        if not times:
            return 1.0
        idx = max(0, min(int(len(times) * percentile) - 1, len(times) - 1))
        return times[idx]

    def filter_by_deadline(self, client_times: dict, deadline: float) -> list:
        return [cid for cid, t in client_times.items() if t <= deadline]

    def compute_straggler_ratio(self, all_client_ids: list, arrived_ids: list) -> float:
        if not all_client_ids:
            return 0.0
        return 1.0 - len(arrived_ids) / len(all_client_ids)

    def is_straggler(self, client_id: int) -> bool:
        return (client_id % self.num_clients) in self.straggler_ids

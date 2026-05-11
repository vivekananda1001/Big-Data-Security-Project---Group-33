import os
import random
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------- Metrics Logging ---------------

METRIC_FIELDS = [
    "round", "global_accuracy", "global_loss",
    "comm_cost_mb",
]


class MetricsLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            writer.writeheader()

    def log(self, metrics: dict):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
            writer.writerow(metrics)


def compute_communication_cost_mb(
    model_size_mb: float, num_clients_per_round: int, num_rounds: int
) -> float:
    return model_size_mb * num_clients_per_round * num_rounds * 2


def find_convergence_round(csv_path: str, threshold: float = 0.80) -> int:
    import pandas as pd
    df = pd.read_csv(csv_path)
    above = df[df["global_accuracy"] >= threshold]
    if above.empty:
        return -1
    return int(above.iloc[0]["round"])


# --------------- Plotting ---------------

def _load_metrics(csv_path: str):
    import pandas as pd
    return pd.read_csv(csv_path)


def plot_accuracy_vs_rounds(csv_paths: dict, output_path: str, title: str = "Global Accuracy vs. Rounds"):
    """csv_paths: {label: csv_path}"""
    plt.figure(figsize=(8, 5))
    for label, path in csv_paths.items():
        df = _load_metrics(path)
        plt.plot(df["round"], df["global_accuracy"], label=label, linewidth=1.5)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy")
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_vs_rounds(csv_paths: dict, output_path: str, title: str = "Global Loss vs. Rounds"):
    plt.figure(figsize=(8, 5))
    for label, path in csv_paths.items():
        df = _load_metrics(path)
        plt.plot(df["round"], df["global_loss"], label=label, linewidth=1.5)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Loss")
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_iid_vs_noniid(iid_csv: str, noniid_csv: str, output_path: str,
                        iid_label: str = "IID", noniid_label: str = "Non-IID (α=0.1)"):
    plt.figure(figsize=(8, 5))
    df_iid = _load_metrics(iid_csv)
    df_noniid = _load_metrics(noniid_csv)
    plt.plot(df_iid["round"], df_iid["global_accuracy"], label=iid_label, linewidth=1.5)
    plt.plot(df_noniid["round"], df_noniid["global_accuracy"], label=noniid_label, linewidth=1.5)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy")
    plt.title("IID vs. Non-IID Comparison")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(csv_paths: dict, output_path: str, metric: str = "global_accuracy"):
    """Bar chart comparing final accuracy across experiments."""
    import pandas as pd

    labels = []
    values = []
    for label, path in csv_paths.items():
        df = pd.read_csv(path)
        labels.append(label)
        values.append(df[metric].iloc[-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Final " + metric.replace("_", " ").title())
    ax.set_title("FedAvg Accuracy Across Heterogeneity Settings")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_results_table(csv_paths: dict, output_path: str):
    """Generate a summary CSV table across all experiments."""
    import pandas as pd

    rows = []
    for label, path in csv_paths.items():
        df = pd.read_csv(path)
        conv_round = find_convergence_round(path)
        final_acc = df["global_accuracy"].iloc[-1] if not df.empty else 0
        final_loss = df["global_loss"].iloc[-1] if not df.empty else 0
        total_rounds = int(df["round"].iloc[-1]) if not df.empty else 0

        rows.append({
            "Experiment": label,
            "Test Accuracy (%)": round(final_acc * 100, 2),
            "Final Loss": round(final_loss, 4),
            "Convergence Round": conv_round,
            "Total Rounds": total_rounds,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    return result_df

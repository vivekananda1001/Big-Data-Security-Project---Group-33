#!/usr/bin/env python3
"""Generate all plots and summary table from FedAvg experiment CSVs."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    plot_accuracy_vs_rounds,
    plot_loss_vs_rounds,
    plot_iid_vs_noniid,
    plot_accuracy_comparison,
    generate_results_table,
)

METRICS_DIR = "results/metrics"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def csv(name):
    return os.path.join(METRICS_DIR, f"{name}.csv")


def exists(name):
    return os.path.exists(csv(name))


def available(names_dict):
    return {k: v for k, v in names_dict.items() if os.path.exists(v)}


# ── Plot 1: Accuracy vs rounds — varying alpha, CIFAR-10 c100 ──
p1 = {
    "α=0.01": csv("fedavg_cifar10_c100_a0.01"),
    "α=0.1": csv("fedavg_cifar10_c100_a0.1"),
    "α=0.5": csv("fedavg_cifar10_c100_a0.5"),
    "α=1.0": csv("fedavg_cifar10_c100_a1.0"),
    "IID": csv("fedavg_cifar10_c100_aiid"),
}
p1 = available(p1)
if p1:
    plot_accuracy_vs_rounds(p1, f"{PLOTS_DIR}/accuracy_vs_alpha_cifar10_c100.png",
                            "FedAvg Accuracy vs. Rounds — CIFAR-10, 100 clients, varying α")
    print("[OK] Plot 1: accuracy_vs_alpha_cifar10_c100.png")

# ── Plot 2: Loss vs rounds — same config ──
if p1:
    plot_loss_vs_rounds(p1, f"{PLOTS_DIR}/loss_vs_alpha_cifar10_c100.png",
                        "FedAvg Loss vs. Rounds — CIFAR-10, 100 clients, varying α")
    print("[OK] Plot 2: loss_vs_alpha_cifar10_c100.png")

# ── Plot 3: IID vs Non-IID — CIFAR-10 ──
if exists("fedavg_cifar10_c100_aiid") and exists("fedavg_cifar10_c100_a0.1"):
    plot_iid_vs_noniid(csv("fedavg_cifar10_c100_aiid"), csv("fedavg_cifar10_c100_a0.1"),
                       f"{PLOTS_DIR}/iid_vs_noniid_cifar10.png",
                       "IID", "Non-IID (α=0.1)")
    print("[OK] Plot 3a: iid_vs_noniid_cifar10.png")

if exists("fedavg_fmnist_c100_aiid") and exists("fedavg_fmnist_c100_a0.1"):
    plot_iid_vs_noniid(csv("fedavg_fmnist_c100_aiid"), csv("fedavg_fmnist_c100_a0.1"),
                       f"{PLOTS_DIR}/iid_vs_noniid_fmnist.png",
                       "IID", "Non-IID (α=0.1)")
    print("[OK] Plot 3b: iid_vs_noniid_fmnist.png")

# ── Plot 4: Accuracy vs rounds — varying client count, CIFAR-10 a0.1 ──
p4 = {
    "10 clients": csv("fedavg_cifar10_c10_a0.1"),
    "50 clients": csv("fedavg_cifar10_c50_a0.1"),
    "100 clients": csv("fedavg_cifar10_c100_a0.1"),
}
p4 = available(p4)
if p4:
    plot_accuracy_vs_rounds(p4, f"{PLOTS_DIR}/accuracy_vs_clients_cifar10_a0.1.png",
                            "FedAvg Accuracy vs. Rounds — CIFAR-10, α=0.1, varying clients")
    print("[OK] Plot 4: accuracy_vs_clients_cifar10_a0.1.png")

# ── Plot 5: Cross-dataset comparison bar chart ──
p5 = {
    "CIFAR-10 α=0.1": csv("fedavg_cifar10_c100_a0.1"),
    "CIFAR-10 IID": csv("fedavg_cifar10_c100_aiid"),
    "FMNIST α=0.1": csv("fedavg_fmnist_c100_a0.1"),
    "FMNIST IID": csv("fedavg_fmnist_c100_aiid"),
    "MNIST α=0.1": csv("fedavg_mnist_c100_a0.1"),
    "MNIST IID": csv("fedavg_mnist_c100_aiid"),
}
p5 = available(p5)
if p5:
    plot_accuracy_comparison(p5, f"{PLOTS_DIR}/accuracy_comparison.png")
    print("[OK] Plot 5: accuracy_comparison.png")

# ── Plot 6: FMNIST accuracy vs rounds — varying alpha ──
p6 = {
    "α=0.1": csv("fedavg_fmnist_c100_a0.1"),
    "α=0.5": csv("fedavg_fmnist_c100_a0.5"),
    "IID": csv("fedavg_fmnist_c100_aiid"),
}
p6 = available(p6)
if p6:
    plot_accuracy_vs_rounds(p6, f"{PLOTS_DIR}/accuracy_vs_alpha_fmnist_c100.png",
                            "FedAvg Accuracy vs. Rounds — FMNIST, 100 clients, varying α")
    print("[OK] Plot 6: accuracy_vs_alpha_fmnist_c100.png")

# ── Summary results table ──
all_csvs = {}
for f in sorted(os.listdir(METRICS_DIR)):
    if f.endswith(".csv"):
        name = f.replace(".csv", "")
        all_csvs[name] = os.path.join(METRICS_DIR, f)

if all_csvs:
    df = generate_results_table(all_csvs, f"{PLOTS_DIR}/summary_table.csv")
    print(f"\n[OK] Summary table: {PLOTS_DIR}/summary_table.csv")
    print(df.to_string(index=False))
else:
    print("[WARN] No CSVs found in results/metrics/")

print("\nDone. All plots saved to results/plots/")

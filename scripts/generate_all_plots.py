#!/usr/bin/env python3
"""Generate all mandatory plots and summary table from experiment CSVs."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    plot_accuracy_vs_rounds,
    plot_loss_vs_rounds,
    plot_iid_vs_noniid,
    plot_fedavg_vs_proposed,
    plot_cat7_metrics,
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


# ── Plot 1: Accuracy vs rounds — all methods, CIFAR-10 c100 a0.1 ──
p1 = {
    "FedAvg": csv("fedavg_cifar10_c100_a0.1"),
    "FedDgc": csv("feddgc_cifar10_c100_a0.1"),
    "FedAsync": csv("fedasync_cifar10_c100_a0.1"),
}
p1 = available(p1)
if p1:
    plot_accuracy_vs_rounds(p1, f"{PLOTS_DIR}/accuracy_vs_rounds_cifar10_c100_a0.1.png",
                            "Global Accuracy vs. Rounds — CIFAR-10, 100 clients, α=0.1")
    print("[OK] Plot 1: accuracy_vs_rounds_cifar10_c100_a0.1.png")

# ── Plot 2: Loss vs rounds — same config ──
if p1:
    plot_loss_vs_rounds(p1, f"{PLOTS_DIR}/loss_vs_rounds_cifar10_c100_a0.1.png",
                        "Global Loss vs. Rounds — CIFAR-10, 100 clients, α=0.1")
    print("[OK] Plot 2: loss_vs_rounds_cifar10_c100_a0.1.png")

# ── Plot 3: IID vs Non-IID — FedAvg ──
if exists("fedavg_cifar10_c100_aiid") and exists("fedavg_cifar10_c100_a0.1"):
    plot_iid_vs_noniid(csv("fedavg_cifar10_c100_aiid"), csv("fedavg_cifar10_c100_a0.1"),
                       f"{PLOTS_DIR}/iid_vs_noniid_fedavg.png",
                       "FedAvg IID", "FedAvg Non-IID (α=0.1)")
    print("[OK] Plot 3a: iid_vs_noniid_fedavg.png")

if exists("feddgc_cifar10_c100_aiid") and exists("feddgc_cifar10_c100_a0.1"):
    plot_iid_vs_noniid(csv("feddgc_cifar10_c100_aiid"), csv("feddgc_cifar10_c100_a0.1"),
                       f"{PLOTS_DIR}/iid_vs_noniid_feddgc.png",
                       "FedDgc IID", "FedDgc Non-IID (α=0.1)")
    print("[OK] Plot 3b: iid_vs_noniid_feddgc.png")

# ── Plot 4: FedAvg vs Proposed — bar chart ──
p4 = {
    "FedAvg CIFAR-10": csv("fedavg_cifar10_c100_a0.1"),
    "FedDgc CIFAR-10": csv("feddgc_cifar10_c100_a0.1"),
    "FedAsync CIFAR-10": csv("fedasync_cifar10_c100_a0.1"),
    "FedAvg FMNIST": csv("fedavg_fmnist_c100_a0.1"),
    "FedDgc FMNIST": csv("feddgc_fmnist_c100_a0.1"),
}
p4 = available(p4)
if p4:
    plot_fedavg_vs_proposed(p4, f"{PLOTS_DIR}/fedavg_vs_proposed.png")
    print("[OK] Plot 4: fedavg_vs_proposed.png")

# ── Plot 5: Category 7 metrics (round time + straggler ratio) ──
p5 = {
    "FedDgc CIFAR-10": csv("feddgc_cifar10_c100_a0.1"),
    "FedAsync CIFAR-10": csv("fedasync_cifar10_c100_a0.1"),
}
p5 = available(p5)
if p5:
    plot_cat7_metrics(p5, f"{PLOTS_DIR}/cat7_metrics.png")
    print("[OK] Plot 5: cat7_metrics.png")

# ── Extra: Accuracy vs rounds for FMNIST ──
pf = {
    "FedAvg": csv("fedavg_fmnist_c100_a0.1"),
    "FedDgc": csv("feddgc_fmnist_c100_a0.1"),
    "FedAsync": csv("fedasync_fmnist_c100_a0.1"),
}
pf = available(pf)
if pf:
    plot_accuracy_vs_rounds(pf, f"{PLOTS_DIR}/accuracy_vs_rounds_fmnist_c100_a0.1.png",
                            "Global Accuracy vs. Rounds — FMNIST, 100 clients, α=0.1")
    print("[OK] Extra: accuracy_vs_rounds_fmnist_c100_a0.1.png")

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

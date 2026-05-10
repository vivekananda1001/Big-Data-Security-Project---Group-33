#!/bin/bash
# Run AFTER all 4 users have finished and CSVs are collected in results/metrics/
# Generates all mandatory plots + summary table

cd "$(dirname "$0")/.."
python scripts/generate_all_plots.py

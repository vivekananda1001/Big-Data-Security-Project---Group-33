#!/bin/bash
# Run AFTER all 4 users have finished and CSVs are collected in results/metrics/
# Generates all mandatory plots + summary table

cd "$(dirname "$0")/.."
source venv/bin/activate
python scripts/generate_all_plots.py

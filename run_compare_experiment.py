"""One-click driver for the retraining-comparison experiment.

Equivalent to running:
    python run.py train --config configs/compare_changed.json
    python compare_runs.py --baseline outputs/compare/baseline --changed outputs/compare/changed --output outputs/compare

How to use (PyCharm, no command line needed):
    right-click this file  ->  Run 'run_compare_experiment'      (~2 minutes)

Edit the channel parameters in configs/compare_changed.json before running.
"""
from __future__ import annotations

import sys

import compare_runs
from elfcomm.config import load_config
from elfcomm.performance import PerformanceTable
from elfcomm.training import train

CHANGED_CONFIG = "configs/compare_changed.json"
BASELINE_DIR = "outputs/compare/baseline"
CHANGED_DIR = "outputs/compare/changed"
OUTPUT_DIR = "outputs/compare"


def main():
    cfg = load_config(CHANGED_CONFIG)
    table = PerformanceTable(cfg)
    print(f"Retraining with channel parameters from {CHANGED_CONFIG} ...", flush=True)
    train(cfg, table)
    print("Training done. Rebuilding the comparison figure ...", flush=True)
    sys.argv = ["compare_runs.py", "--baseline", BASELINE_DIR,
                "--changed", CHANGED_DIR, "--output", OUTPUT_DIR]
    compare_runs.main()
    print(f"Finished. Final figure: {OUTPUT_DIR}/retrain_comparison.png", flush=True)


if __name__ == "__main__":
    main()

"""One-click driver for the mid-run channel-change experiment.

Trains ONE run of 1000 episodes: the initial channel parameters are used until
episode N (see configs/shift_experiment.json -> "channel_shift"), then the channel
parameters are switched and training continues, producing a single
"high -> stable -> jump -> re-stabilise" BER curve plus the +/-15% criterion check.

Equivalent to running:
    python run.py train --config configs/shift_experiment.json
    python plot_shift_recovery.py --directory outputs/shift

How to use (PyCharm, no command line needed):
    right-click this file  ->  Run 'run_shift_experiment'      (~4-5 minutes)

Edit configs/shift_experiment.json to change the shift episode or the new channel.
"""
from __future__ import annotations

import sys

import plot_shift_recovery
from elfcomm.config import load_config
from elfcomm.performance import PerformanceTable
from elfcomm.training import train

CONFIG = "configs/shift_experiment.json"


def main():
    cfg = load_config(CONFIG)
    output_dir = cfg["output_dir"]
    shift = cfg["channel_shift"]
    table = PerformanceTable(cfg)
    print(f"Training {cfg['training']['episodes']} episodes; channel changes at episode "
          f"{shift['episode']} -> {shift['environment']}", flush=True)
    train(cfg, table)
    print("Training done. Building the recovery figure ...", flush=True)
    sys.argv = ["plot_shift_recovery.py", "--directory", output_dir, "--output", output_dir]
    plot_shift_recovery.main()
    print(f"Finished. Final figure: {output_dir}/shift_recovery.png", flush=True)


if __name__ == "__main__":
    main()

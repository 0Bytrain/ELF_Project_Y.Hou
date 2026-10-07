"""Compare two training runs (baseline channel vs changed channel) and check
whether the stabilised BER stays within a relative tolerance band (+/-15%).

Usage:
    python compare_runs.py --baseline outputs/compare/baseline ^
                           --changed  outputs/compare/changed ^
                           --output   outputs/compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from elfcomm.config import save_json
from elfcomm.plots import column, ema, read_csv


def stable_level(directory, window):
    """Mean of the last `window` greedy-validation BER values of a run."""
    rows = read_csv(Path(directory) / "validation_history.csv")
    ber = column(rows, "ber_estimate")
    return float(np.nanmean(ber[-window:]))


def run_label(directory):
    path = Path(directory) / "run_config.json"
    if not path.exists():
        return Path(directory).name
    cfg = json.loads(path.read_text(encoding="utf-8"))
    env = cfg.get("fixed_environment") or (cfg.get("channel_shift") or {}).get("before_environment") or {}
    noise = env.get("environment_noise_asd_pt_rt_hz")
    return f"noise ASD {noise:g} pT/sqrt(Hz)" if noise is not None else Path(directory).name


def main():
    parser = argparse.ArgumentParser(description="Retraining comparison after a channel-parameter change")
    parser.add_argument("--baseline", required=True, help="directory of the baseline training run")
    parser.add_argument("--changed", required=True, help="directory of the changed-channel training run")
    parser.add_argument("--output", required=True, help="directory for the comparison figure and json")
    parser.add_argument("--window", type=int, default=3, help="how many final validation points define the stable level")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed relative difference, 0.15 = +/-15%%")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_stable = stable_level(args.baseline, args.window)
    changed_stable = stable_level(args.changed, args.window)
    ratio = changed_stable / baseline_stable
    within = abs(ratio - 1.0) <= args.tolerance

    fig, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for directory, color in ((args.baseline, "#176B9B"), (args.changed, "#D46B27")):
        label = run_label(directory)
        rows = read_csv(Path(directory) / "training_history.csv")
        axes.semilogy(column(rows, "episode"), ema(column(rows, "ber_estimate"), 0.1),
                      color=color, linewidth=1.8, label=f"training EMA — {label}")
        validation = read_csv(Path(directory) / "validation_history.csv")
        axes.semilogy(column(validation, "episode"), column(validation, "ber_estimate"),
                      "o", color=color, markersize=3, alpha=0.85, label=f"greedy validation — {label}")
    axes.axhspan(baseline_stable * (1 - args.tolerance), baseline_stable * (1 + args.tolerance),
                 color="#176B9B", alpha=0.12, label=f"+/-{args.tolerance:.0%} band of baseline")
    axes.axhline(baseline_stable, color="#176B9B", linestyle="--", linewidth=1.2,
                 label=f"baseline stable ~= {baseline_stable:.3e}")
    axes.set(xlabel="Training episode (adaptation process)",
             ylabel="Post-Polar payload BER estimate",
             title="Retraining after channel-parameter change: stabilised BER comparison")
    axes.grid(which="both", alpha=0.2)
    axes.legend(fontsize=8)
    axes.annotate(f"baseline stable ~= {baseline_stable:.3e}\nchanged stable ~= {changed_stable:.3e}\n"
                  f"ratio = {ratio:.3f}   criterion |ratio-1| <= {args.tolerance:g}: "
                  f"{'PASS' if within else 'FAIL'}",
                  xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.savefig(output / "retrain_comparison.png", dpi=180)
    plt.close(fig)
    save_json(output / "comparison_summary.json", {
        "window": args.window, "tolerance": args.tolerance,
        "baseline_stable_ber": baseline_stable, "changed_stable_ber": changed_stable,
        "ratio": ratio, "criterion": "abs(changed/baseline - 1) <= tolerance",
        "criterion_met": bool(within),
    })
    print(f"baseline={baseline_stable:.3e}  changed={changed_stable:.3e}  ratio={ratio:.3f}  "
          f"{'PASS' if within else 'FAIL'}")


if __name__ == "__main__":
    main()

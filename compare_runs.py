"""Compare two training runs (baseline channel vs changed channel) and check
whether the stabilised BER stays within a relative tolerance band (+/-15%).

The figure follows the same style as outputs/ber_vs_episode.png:
two stacked panels (3:1, shared x axis), DejaVu Sans, dpi=180.

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

BLUE = "#176B9B"
ORANGE = "#D46B27"
RED = "#AD3434"


def stable_level(directory, window):
    """Mean of the last `window` greedy-validation BER values of a run."""
    rows = read_csv(Path(directory) / "validation_history.csv")
    return float(np.nanmean(column(rows, "ber_estimate")[-window:]))


def run_label(directory):
    path = Path(directory) / "run_config.json"
    if not path.exists():
        return Path(directory).name
    cfg = json.loads(path.read_text(encoding="utf-8"))
    noise = (cfg.get("fixed_environment") or {}).get("environment_noise_asd_pt_rt_hz")
    return f"noise ASD {noise:g} pT/sqrt(Hz)" if noise is not None else Path(directory).name


def main():
    parser = argparse.ArgumentParser(description="Retraining comparison after a channel-parameter change")
    parser.add_argument("--baseline", required=True, help="directory of the baseline training run")
    parser.add_argument("--changed", required=True, help="directory of the changed-channel training run")
    parser.add_argument("--output", required=True, help="directory for the comparison figure and json")
    parser.add_argument("--window", type=int, default=3, help="how many final validation points define the stable level")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed relative difference, 0.15 = +/-15%%")
    parser.add_argument("--target-ber", type=float, default=1e-4)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_stable = stable_level(args.baseline, args.window)
    changed_stable = stable_level(args.changed, args.window)
    ratio = changed_stable / baseline_stable
    within = abs(ratio - 1.0) <= args.tolerance

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    top, bottom = axes
    runs = [(args.baseline, BLUE, run_label(args.baseline)),
            (args.changed, ORANGE, run_label(args.changed))]
    for directory, color, label in runs:
        rows = read_csv(Path(directory) / "training_history.csv")
        x, ber = column(rows, "episode"), column(rows, "ber_estimate")
        top.semilogy(x, ber, color=color, alpha=0.18, linewidth=0.8)
        top.semilogy(x, ema(ber, 0.1), color=color, linewidth=1.8, label=f"training EMA — {label}")
        validation = read_csv(Path(directory) / "validation_history.csv")
        xv, bv = column(validation, "episode"), column(validation, "ber_estimate")
        top.semilogy(xv, bv, "o--", color=color, markersize=4, linewidth=0.9,
                     label=f"greedy validation — {label}")
        bottom.plot(xv, bv / baseline_stable * 100.0 - 100.0, "o--", color=color,
                    markersize=4, linewidth=0.9, label=f"greedy validation — {label}")
    top.axhspan(baseline_stable * (1 - args.tolerance), baseline_stable * (1 + args.tolerance),
                color=BLUE, alpha=0.10, label=f"+/-{args.tolerance:.0%} band of baseline")
    top.axhline(baseline_stable, color=BLUE, linestyle="--", linewidth=1.1,
                label=f"baseline stable ~ {baseline_stable:.2e}")
    top.axhline(changed_stable, color=ORANGE, linestyle="--", linewidth=1.1,
                label=f"changed stable ~ {changed_stable:.2e}")
    top.axhline(args.target_ber, color=RED, linestyle=":", linewidth=1.1,
                label=f"Target BER {args.target_ber:g}")
    top.set(ylabel="Post-Polar payload BER estimate",
            title="BER versus training episodes — retraining after channel-parameter change")
    top.grid(which="both", alpha=0.18)
    top.legend(fontsize=8, loc="upper right")
    top.annotate(f"baseline stable ~= {baseline_stable:.3e}\nchanged stable ~= {changed_stable:.3e}\n"
                 f"ratio = {ratio:.3f}   criterion |ratio-1| <= {args.tolerance:g}: "
                 f"{'PASS' if within else 'FAIL'}",
                 xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    bottom.axhline(0.0, color="#666666", linewidth=1.0)
    bottom.axhline(+args.tolerance * 100, color=RED, linestyle="--", linewidth=1.0,
                   label=f"+{args.tolerance:.0%} criterion")
    bottom.axhline(-args.tolerance * 100, color=RED, linestyle="--", linewidth=1.0,
                   label=f"-{args.tolerance:.0%} criterion")
    bottom.set(xlabel="Training episode (adaptation process)",
               ylabel="Validation BER vs baseline (%)")
    bottom.grid(alpha=0.2)
    bottom.legend(fontsize=8, loc="upper right")
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

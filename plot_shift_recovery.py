"""Plot a single 1000-episode training run with a mid-run channel-parameter change.

Reads <directory>/validation_history.csv and <directory>/run_config.json, draws the
BER-versus-episode trajectory (greedy validation only, one clean curve) with the
channel-change episode marked, a +/-15% criterion band referenced to the pre-shift
stable level, and a deviation panel. Writes shift_recovery.png + shift_summary.json.

Style follows outputs/compare/retrain_comparison.png (two stacked panels 3:1,
DejaVu Sans, dpi=180).

Usage:
    python plot_shift_recovery.py --directory outputs/shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from elfcomm.config import save_json
from elfcomm.plots import column, read_csv

BLUE = "#176B9B"
ORANGE = "#D46B27"
RED = "#AD3434"
GRAY = "#666666"


def stable_level(rows, window, episodes_up_to=None):
    """Mean of the last `window` validation BERs (optionally only up to an episode)."""
    selected = rows
    if episodes_up_to is not None:
        selected = [row for row in rows if float(row["episode"]) <= episodes_up_to]
    return float(np.nanmean(column(selected, "ber_estimate")[-window:]))


def main():
    parser = argparse.ArgumentParser(description="Mid-run channel-change recovery figure")
    parser.add_argument("--directory", required=True, help="training output directory")
    parser.add_argument("--output", default=None, help="directory for figure and json (defaults to --directory)")
    parser.add_argument("--window", type=int, default=3, help="how many validation points define a stable level")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed relative difference, 0.15 = +/-15%")
    parser.add_argument("--target-ber", type=float, default=1e-4)
    args = parser.parse_args()

    directory = Path(args.directory)
    output = Path(args.output) if args.output else directory
    output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(directory / "validation_history.csv")
    config_path = directory / "run_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    shift = cfg.get("channel_shift") or {}
    episodes_all = column(rows, "episode")
    shift_episode = int(shift.get("episode", episodes_all[len(episodes_all) // 2]))

    pre_stable = stable_level(rows, args.window, episodes_up_to=shift_episode)
    post_stable = stable_level(rows, args.window)
    ratio = post_stable / pre_stable
    within = abs(ratio - 1.0) <= args.tolerance

    episodes = column(rows, "episode")
    ber = column(rows, "ber_estimate")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    top, bottom = axes

    top.semilogy(episodes, ber, "o--", color=BLUE, markersize=4, linewidth=1.1,
                 label="greedy validation BER")
    top.axvline(shift_episode, color=GRAY, linestyle="-.", linewidth=1.1,
                label=f"channel changed @ episode {shift_episode}")
    top.axhspan(pre_stable * (1 - args.tolerance), pre_stable * (1 + args.tolerance),
                color=BLUE, alpha=0.12, label=f"+/-{args.tolerance:.0%} band of pre-shift stable")
    top.axhline(pre_stable, color=BLUE, linestyle="--", linewidth=1.1,
                label=f"pre-shift stable ~ {pre_stable:.2e}")
    top.axhline(post_stable, color=ORANGE, linestyle="--", linewidth=1.1,
                label=f"post-shift stable ~ {post_stable:.2e}")
    top.axhline(args.target_ber, color=RED, linestyle=":", linewidth=1.1,
                label=f"Target BER {args.target_ber:g}")
    top.set(ylabel="Post-Polar payload BER estimate",
            title="BER versus training episodes — mid-run channel-parameter change\n"
                  f"stabilised BER: before {pre_stable:.2e} -> after {post_stable:.2e}"
                  f"   |   ratio {ratio:.3f}   |   criterion ±{args.tolerance:.0%}: "
                  f"{'PASS' if within else 'FAIL'}")
    top.grid(which="both", alpha=0.18)
    peak = float(np.nanmax(ber)) if np.any(np.isfinite(ber)) else 3e-4
    top.set_ylim(8e-6, max(3e-4, 1.4 * peak))  # frame the initial high, the jump and the floor
    top.legend(fontsize=8, loc="upper right", framealpha=0.9)

    bottom.axvline(shift_episode, color=GRAY, linestyle="-.", linewidth=1.0)
    bottom.plot(episodes, ber / pre_stable * 100.0 - 100.0, "o--", color=BLUE,
                markersize=4, linewidth=1.0, label="greedy validation BER vs pre-shift stable")
    bottom.axhline(0.0, color=GRAY, linewidth=1.0)
    bottom.axhline(+args.tolerance * 100, color=RED, linestyle="--", linewidth=1.0,
                   label=f"±{args.tolerance:.0%} criterion")
    bottom.axhline(-args.tolerance * 100, color=RED, linestyle="--", linewidth=1.0)
    bottom.set(xlabel="Training episode (adaptation process)",
               ylabel="BER vs pre-shift stable (%)")
    bottom.yaxis.labelpad = 14  # push the long label left so it clears the upper panel's tick
    bottom.set_ylim(-30, 80)  # zoom on the criterion region; early transients go off-scale
    bottom.grid(alpha=0.2)
    bottom.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.savefig(output / "shift_recovery.png", dpi=180)
    plt.close(fig)
    save_json(output / "shift_summary.json", {
        "window": args.window, "tolerance": args.tolerance,
        "shift_episode": shift_episode,
        "pre_shift_stable_ber": pre_stable, "post_shift_stable_ber": post_stable,
        "ratio": ratio, "criterion": "abs(post/pre - 1) <= tolerance",
        "criterion_met": bool(within),
    })
    print(f"pre_shift={pre_stable:.3e}  post_shift={post_stable:.3e}  ratio={ratio:.3f}  "
          f"{'PASS' if within else 'FAIL'}")


if __name__ == "__main__":
    main()

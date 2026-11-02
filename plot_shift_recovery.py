"""Plot observed training/adaptation with one absolute BER target in both phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elfcomm.config import save_json
from elfcomm.convergence import phase_assessment, tail_statistics
from elfcomm.plots import (PREFERRED_POWER_TICKS, ber_plot_values, column,
                           experiment_label, phase_ema, power_ticks, read_csv,
                           set_ber_limits)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def convergence_window(rows, window, episodes_up_to=None, episodes_from=None,
                       key="ber_estimate", tolerance=0.15):
    return tail_statistics(rows, window, tolerance, key, episodes_up_to, episodes_from)


def figure_filename(cfg):
    return "training_ber_curve.png"


def plot_clean_ber(ax, rows, alpha, shift, target):
    """Plot one smoothed blue BER curve; zero-error samples use their 1/N limit."""
    displayed, _ = ber_plot_values(rows)
    display_rows = [{**row, "ber_estimate": value}
                    for row, value in zip(rows, displayed)]
    smooth = phase_ema(display_rows, "ber_estimate", alpha, shift)
    ax.semilogy(column(rows, "episode"), smooth, color="#176B9B", linewidth=1.5,
                label="Training BER variation")
    set_ber_limits(ax, rows, target)
    if shift is not None:
        ax.axvline(shift + 0.5, color="#666666", linestyle=":",
                   label="Environment change")
    ax.set_ylabel("Training BER estimate")
    ax.grid(which="major", alpha=0.18)


def plot_clean_power(ax, rows, alpha, shift):
    ax.plot(column(rows, "episode"),
            phase_ema(rows, "mean_tx_power_kw", alpha, shift),
            color="#176B9B", linewidth=1.6, label="training mean power")
    if shift is not None:
        ax.axvline(shift + 0.5, color="#666666", linestyle=":")
    ax.set_ylabel("TX power (kW)")
    ax.grid(alpha=0.18)


def final_stable(rows, shift, assessment, window):
    selected = [r for r in rows
                if r.get("evaluation_event") not in ("before_update", "untrained")]
    post = [r for r in selected if float(r["episode"]) > shift]
    stats = assessment.get("post", {})
    start, end = stats.get("start"), stats.get("end")
    if start is None or end is None:
        tail = post[-window:]
    else:
        tail = [r for r in post if start <= float(r["episode"]) <= end]
    if not tail:
        return None
    displayed, zeros = ber_plot_values(tail)
    positive = displayed[np.isfinite(displayed) & (displayed > 0)]
    if not positive.size:
        return None
    snr = column(tail, "mean_snr_db")
    snr = snr[np.isfinite(snr)]
    esn0 = column(tail, "mean_esn0_db")
    esn0 = esn0[np.isfinite(esn0)]
    return {
        "level": float(np.median(positive)),
        "zero_errors": bool(np.all(zeros)),
        "snr_db": float(np.mean(snr)) if snr.size else float("nan"),
        "esn0_db": float(np.mean(esn0)) if esn0.size else float("nan"),
        "start": float(tail[0]["episode"]),
        "end": float(tail[-1]["episode"]),
    }


def final_stable_text(info):
    if info is None:
        return "Final stable BER: n/a"
    level = (f"≤{info['level']:.3e}" if info["zero_errors"]
             else f"{info['level']:.3e}")
    lines = [f"Final stable BER: {level}"]
    if np.isfinite(info["snr_db"]):
        line = f"SNR: {info['snr_db']:.2f} dB"
        if np.isfinite(info["esn0_db"]):
            line += f" (Es/N0: {info['esn0_db']:.2f} dB)"
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--convergence-tolerance", type=float, default=0.15,
                        help="Relative spread of the final window; not the BER target")
    parser.add_argument("--target-ber", type=float, default=None)
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    output = Path(args.output or directory)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    target = cfg["objective"]["target_ber"] if args.target_ber is None else args.target_ber
    shift = int(cfg["channel_shift"]["episode"])
    rows = read_csv(directory / "validation_history.csv")
    history = read_csv(directory / "training_history.csv")
    stability = cfg["training"].get("convergence")
    window = args.window or (stability or {}).get("window", 10)
    assessment = phase_assessment(rows, shift, target, window, args.convergence_tolerance, stability)
    final = final_stable(rows, shift, assessment, window)
    ordinary = [r for r in rows if r.get("evaluation_event") != "before_update"]
    jump = [r for r in rows if r.get("evaluation_event") == "before_update"]

    def phase_label(phase):
        stats = assessment[phase]
        if not stats["converged"]:
            return "not established"
        return "estimated optimum stable" if stats["resolved"] else "estimated optimum stable; BER unresolved"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    top, bottom = axes
    plot_clean_ber(top, history, cfg["training"]["ema_alpha"], shift, target)
    if final is not None:
        top.text(0.02, 0.04, final_stable_text(final), transform=top.transAxes,
                 ha="left", va="bottom", fontsize=8,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                       "edgecolor": "#CCCCCC", "alpha": 0.9})
    scenario = experiment_label(cfg)
    top.set(title=scenario or "BER versus training episode")
    top.legend(fontsize=8, loc="upper right")
    plot_clean_power(bottom, history, cfg["training"]["ema_alpha"], shift)
    bottom.set(xlabel="Training episode", ylabel="TX power (kW)")
    power_values = cfg["actions"]["tx_power_kw"]
    bottom.set_yscale("log")
    bottom.set_ylim(min(power_values) * 0.9, max(power_values) * 1.1)
    bottom.set_yticks(power_ticks(power_values, preferred=PREFERRED_POWER_TICKS))
    bottom.tick_params(axis="y", labelsize=7, pad=2)
    bottom.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    bottom.legend(fontsize=7, loc="upper right")
    fig.savefig(output / figure_filename(cfg), dpi=180)
    plt.close(fig)
    save_json(output / "shift_summary.json", {
        "schema": 5, "main_curve": "raw exploratory training BER and phase-reset EMA",
        "policy_assessment_source": "unfiltered fixed-scenario evaluations",
        "deployment_file": "deployment_performance.png", "target_ber": target, "criterion": ("raw fixed evaluations: all frames meet estimated target, power/reward/configuration stable, model minimum power verified; exploration finished when required; BER precision reported separately" if stability else "legacy BER stability"),
        "training_actions_file": "training_actions.csv" if (directory / "training_actions.csv").exists() else None,
        "legacy_ber_stability_relative_tolerance": args.convergence_tolerance, "window": window,
        "final_stable": final,
        "smoothing": "causal phase-reset EMA for display only; not used in assessment",
        "shift_episode": shift, "controlled_experiment": cfg.get("controlled_experiment"),
        "before_environment": cfg["fixed_environment"],
        "after_environment": cfg["channel_shift"]["environment"],
        "within_episode_drift": cfg["training"].get("dynamic_environment", True),
        **assessment,
        "ber_source": "empirical_monotone_linear_table_with_zero_count_resolution_flags",
        "reliability_certified": False,
    })
    print(f"Absolute BER target={target:g}; pre={phase_label('pre')}; post={phase_label('post')}; "
          f"stability + target={assessment['criterion_met']}")
    print(final_stable_text(final))


if __name__ == "__main__":
    main()

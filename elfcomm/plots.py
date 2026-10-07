from __future__ import annotations

import csv
from pathlib import Path

from .config import save_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ema(values, alpha):
    output = np.full(len(values), np.nan)
    previous = np.nan
    for i, value in enumerate(values):
        if np.isfinite(value):
            previous = value if not np.isfinite(previous) else alpha * value + (1 - alpha) * previous
            output[i] = previous
    return output


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    return np.array([float(row[name]) if row.get(name) else np.nan for row in rows])


def plot_training(output, alpha=0.1, target_ber=1e-4):
    output = Path(output)
    train = read_csv(output / "training_history.csv")
    validation = read_csv(output / "validation_history.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    x, xv = column(train, "episode"), column(validation, "episode")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = [("mean_reward", "Mean reward"), ("loss", "TD loss"),
                      ("goodput_bps_estimate", "Estimated goodput (bit/s)"),
                      ("ber_target_met_fraction_estimate", "Estimated BER target met fraction")]
    for ax, (key, label) in zip(axes.flat, specifications):
        values = column(train, key)
        ax.plot(x, values, color="#176B9B", alpha=0.2, linewidth=0.8)
        ax.plot(x, ema(values, alpha), color="#176B9B", linewidth=1.8, label="DQN training EMA")
        if key != "loss":
            ax.plot(xv, column(validation, key), "o--", color="#D46B27", markersize=3, label="DQN greedy validation")
        ax.set(xlabel="Training episode", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Tensor DQN convergence (relative-link simulation)")
    fig.savefig(output / "training_convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    values = column(train, "ber_estimate")
    axes[0].semilogy(x, values, color="#176B9B", alpha=0.2, linewidth=0.8)
    axes[0].semilogy(x, ema(values, alpha), color="#176B9B", linewidth=1.8, label="DQN training BER estimate (EMA)")
    axes[0].semilogy(xv, column(validation, "ber_estimate"), "o--", color="#D46B27", markersize=4,
                     label="DQN greedy validation BER estimate")
    axes[0].axhline(target_ber, color="#AD3434", linestyle=":", label=f"Target {target_ber:g}")
    axes[0].set(ylabel="Post-Polar payload BER estimate", title="BER versus training episodes — finite-sample table estimates")
    axes[0].legend(fontsize=9)
    axes[0].grid(which="both", alpha=0.18)
    axes[1].plot(x, column(train, "outage_fraction"), label="No legal action / no transmission", color="#666666")
    axes[1].set(xlabel="Training episode", ylabel="Outage fraction", ylim=(-0.03, 1.03))
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.savefig(output / "ber_vs_episode.png", dpi=180)
    plot_action_iterations(output)
    plt.close(fig)


def plot_action_iterations(output):
    output = Path(output)
    rows = read_csv(output / "training_history.csv")
    if not rows or "greedy_frequency_hz" not in rows[0]:
        return
    episodes = column(rows, "episode")
    modulation_map = {"BPSK": 0, "QPSK": 1, "MSK": 2}
    modulation = np.array([modulation_map.get(row.get("greedy_modulation", ""), np.nan) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = [("greedy_frequency_hz", "Selected frequency (Hz)", None),
              ("greedy_modulation", "Selected modulation", (modulation, ["BPSK", "QPSK", "MSK"])),
              ("greedy_code_rate", "Selected Polar code rate", None),
              ("greedy_symbol_rate", "Selected symbol rate (sym/s)", None)]
    for ax, (key, label, special) in zip(axes.flat, series):
        values = special[0] if special else column(rows, key)
        ax.plot(episodes, values, color="#176B9B", linewidth=1.4, drawstyle="steps-post")
        ax.set(xlabel="Training episode", ylabel=label)
        if special:
            ax.set_yticks(range(len(special[1])), special[1])
        ax.grid(alpha=0.2)
    fig.suptitle("DQN four-parameter decision versus training episodes")
    fig.savefig(output / "decision_variables_vs_episode.png", dpi=180)
    plt.close(fig)

def plot_table(table, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, modulation in zip(axes, table.cfg["actions"]["modulation"]):
        for rate in table.cfg["actions"]["code_rate"]:
            ber, _ = table.curves[(modulation, rate)]
            ax.semilogy(table.knots, ber, "o-", markersize=3, label=f"Nominal R={rate:.3f}")
        ax.set(xlabel="Es/N0 (dB)", ylabel="Smoothed payload BER estimate", title=modulation)
        ax.grid(which="both", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Real Polar Monte Carlo table — zero-error points retain sampling uncertainty")
    fig.savefig(Path(output) / "ber_table.png", dpi=180)
    plt.close(fig)


def plot_channel_shift_recovery(output, shift_episode, cfg=None):
    """BER versus training episode around a mid-training channel-parameter change."""
    output = Path(output)
    cfg = cfg or {}
    rows = read_csv(output / "training_history.csv")
    validation = read_csv(output / "validation_history.csv")
    alpha = cfg.get("training", {}).get("ema_alpha", 0.1)
    target_ber = cfg.get("objective", {}).get("target_ber", 1e-4)
    window = int(cfg.get("channel_shift", {}).get("recovery_window", 3))
    episode = column(rows, "episode")
    ber = column(rows, "ber_estimate")
    validation_episode = column(validation, "episode")
    validation_ber = column(validation, "ber_estimate")
    before = validation_ber[validation_episode < shift_episode][-window:]
    after = validation_ber[validation_episode > shift_episode][-window:]
    baseline = float(np.nanmean(before)) if before.size else float("nan")
    final = float(np.nanmean(after)) if after.size else float("nan")
    ratio = final / baseline if np.isfinite(baseline) and np.isfinite(final) and baseline > 0 else float("nan")
    fig, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axes.semilogy(episode, ber, color="#176B9B", alpha=0.25, linewidth=0.8, label="Per-episode BER estimate")
    axes.semilogy(episode, ema(ber, alpha), color="#176B9B", linewidth=2.0, label="Training EMA")
    axes.semilogy(validation_episode, validation_ber, "o--", color="#D46B27", markersize=4, label="Greedy validation")
    axes.axvline(shift_episode, color="#AD3434", linestyle="--", linewidth=1.6,
                 label=f"Channel parameters changed (episode {shift_episode})")
    axes.axhline(target_ber, color="#666666", linestyle=":", label=f"Target BER {target_ber:g}")
    axes.set(xlabel="Training episode (adaptation process)", ylabel="Post-Polar payload BER estimate",
             title="DQN adaptation after channel-parameter change")
    axes.grid(which="both", alpha=0.2)
    axes.legend(fontsize=9)
    if np.isfinite(ratio):
        axes.annotate(f"pre-shift baseline ~= {baseline:.2e}\nfinal stable ~= {final:.2e}\n"
                      f"final / baseline = {ratio:.2f}  (criterion <= 1.15)",
                      xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.savefig(output / "adaptation_after_channel_shift.png", dpi=180)
    plt.close(fig)
    save_json(output / "adaptation_summary.json", {
        "shift_episode": int(shift_episode), "recovery_window": window,
        "baseline_mean_ber": baseline, "final_mean_ber": final, "ratio": ratio,
        "criterion": "final_mean_ber / baseline_mean_ber <= 1.15",
        "criterion_met": bool(np.isfinite(ratio) and ratio <= 1.15),
    })

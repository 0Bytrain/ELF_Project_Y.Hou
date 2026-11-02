from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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
    return np.array([float(row[name]) if row.get(name) not in (None, "") else np.nan for row in rows])


PREFERRED_POWER_TICKS = (5, 20, 40, 100, 250, 500)


def power_ticks(values, maximum=10, preferred=None):
    """Choose tick labels from the configured power tiers, never from LINSPACE."""
    ordered = sorted(set(float(value) for value in values))
    if not ordered:
        return []
    if preferred is not None:
        selected = [float(value) for value in preferred
                    if any(abs(float(value) - tier) < 1e-9 for tier in ordered)]
        if selected:
            return selected
    if len(ordered) <= maximum:
        return ordered
    targets = np.linspace(ordered[0], ordered[-1], maximum)
    selected = []
    for target in targets:
        nearest = min(ordered, key=lambda value: abs(value - target))
        if nearest not in selected:
            selected.append(nearest)
    return sorted(selected)


def ber_plot_values(rows, key="ber_estimate", resolution_key="ber_resolution_limit"):
    values = column(rows, key)
    zeros = np.isfinite(values) & (values == 0)
    resolution = column(rows, resolution_key)
    displayed = values.copy()
    displayed[zeros] = np.where(np.isfinite(resolution[zeros]) & (resolution[zeros] > 0),
                                resolution[zeros], np.nan)
    return displayed, zeros


def plot_ber_series(ax, rows, key="ber_estimate", xkey="episode", color="#176B9B",
                    label="BER estimate", resolution_key="ber_resolution_limit", **kwargs):
    displayed, zeros = ber_plot_values(rows, key, resolution_key)
    x = column(rows, xkey)
    # Zero-count observations have a separate marker. Their display positions
    # must never enter smoothing, threshold checks or convergence statistics.
    ax.semilogy(x, np.where(zeros, np.nan, displayed), color=color, label=label, **kwargs)
    if zeros.any():
        ax.scatter(x[zeros], displayed[zeros], facecolors="none", edgecolors=color,
                   marker="v", s=22, label="Zero errors observed: shown at 1/N (display only)")
    return displayed


def phase_ema(rows, key, alpha, shift_episode=None):
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must lie in (0, 1]")
    values = column(rows, key)
    result = np.full(len(rows), np.nan)
    phases = column(rows, "episode") > shift_episode if shift_episode is not None else np.zeros(len(rows), dtype=bool)
    for phase in (False, True):
        indices = np.flatnonzero(phases == phase)
        result[indices] = ema(values[indices], alpha)
    return result


def plot_policy_ber(ax, rows, alpha=0.2, shift_episode=None, target_ber=None):
    x = column(rows, "episode")
    values = column(rows, "ber_estimate")
    logarithmic = target_ber is not None and target_ber < 1e-3
    displayed, zeros = ber_plot_values(rows)
    plotted = np.where(zeros, np.nan, values) if logarithmic else values
    smooth = phase_ema(rows, "ber_estimate", alpha, shift_episode)
    smooth_plot = np.where(smooth > 0, smooth, np.nan) if logarithmic else smooth
    resolution = column(rows, "ber_resolution_limit")
    if logarithmic:
        smooth_plot[(values == 0) & np.isfinite(resolution) & (smooth < resolution)] = np.nan
    if logarithmic:
        ax.set_yscale("log")
    phases = x > shift_episode if shift_episode is not None else np.zeros(len(rows), dtype=bool)
    for phase in (False, True):
        selected = phases == phase
        if not selected.any():
            continue
        ax.plot(x[selected], plotted[selected], "o-", color="#86B1CC", alpha=0.55, linewidth=0.7,
                markersize=2.3, label="Fixed-scenario evaluation (raw)" if not phase else "_nolegend_")
        ax.plot(x[selected], smooth_plot[selected], color="#176B9B", linewidth=2.0,
                label="Evaluation BER EMA (display only)" if not phase else "_nolegend_")
    zeros = np.isfinite(values) & (values == 0)
    if zeros.any():
        ax.scatter(x[zeros], displayed[zeros] if logarithmic else values[zeros], marker="v", s=22, facecolors="none",
                   edgecolors="#176B9B", label="Zero errors: shown at 1/N, unresolved" if logarithmic else "Zero errors in table; true BER unresolved")
    if not logarithmic:
        ax.set_ylim(bottom=0)
    return smooth


def plot_learning_ber(ax, rows, alpha, shift_episode, target_ber):
    plot_ber_series(ax, rows, alpha=0.22, linewidth=0.65,
                    label="Training BER with exploration (raw)")
    smooth = phase_ema(rows, "ber_estimate", alpha, shift_episode)
    displayed_smooth = np.where(smooth > 0, smooth, np.nan)
    raw = column(rows, "ber_estimate")
    resolution = column(rows, "ber_resolution_limit")
    displayed_smooth[(raw == 0) & np.isfinite(resolution) & (smooth < resolution)] = np.nan
    x = column(rows, "episode")
    phases = x > shift_episode if shift_episode is not None else np.zeros(len(rows), dtype=bool)
    for phase in (False, True):
        selected = phases == phase
        ax.semilogy(x[selected], displayed_smooth[selected],
                    color="#176B9B", linewidth=1.9,
                    label="Training BER EMA (display only)" if not phase else "_nolegend_")
    ax.axhline(target_ber, color="#AD3434", linestyle="--", label=f"Final policy BER target {target_ber:g}")
    set_ber_limits(ax, rows, target_ber)
    if shift_episode is not None:
        ax.axvline(shift_episode + 0.5, color="#666666", linestyle=":", label="Environment change")
    ax.set_ylabel("Payload BER estimate")
    ax.grid(which="major", alpha=0.18)
    ax.legend(fontsize=8)
    return smooth


def plot_power_comparison(ax, rows, alpha=0.1, shift_episode=None, action_rows=None):
    episodes = column(rows, "episode")
    if action_rows:
        ax.scatter(column(action_rows, "episode_position"), column(action_rows, "tx_power_kw"),
                   color="#76A9CE", alpha=0.12, s=3, rasterized=True,
                   label="Actual training actions (exploration included)")
    sampled = column(rows, "mean_tx_power_kw")
    if np.isfinite(sampled).any():
        ax.plot(episodes, sampled, color="#76A9CE", alpha=0.3, linewidth=0.7,
                label="Training power: episode mean (exploration included)")
        ax.plot(episodes, phase_ema(rows, "mean_tx_power_kw", alpha, shift_episode),
                color="#176B9B", linewidth=1.6, label="Training mean power EMA (display only)")
    ax.set_ylabel("TX power (kW)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.legend(fontsize=7, ncol=1)


def set_ber_limits(ax, rows, target_ber):
    displayed, _ = ber_plot_values(rows)
    positive = displayed[np.isfinite(displayed) & (displayed > 0)]
    if target_ber < 1e-3:
        ax.set_yscale("log")
        lower = min(target_ber / 10, positive.min() / 1.6) if positive.size else target_ber / 10
        upper = max(target_ber * 3, positive.max() * 1.8) if positive.size else target_ber * 3
        ax.set_ylim(max(lower, np.finfo(float).tiny), upper)
    else:
        ax.set_ylim(0, max(target_ber * 1.15, positive.max() * 1.1 if positive.size else 0))


def experiment_label(cfg):
    experiment = cfg.get("controlled_experiment")
    if not experiment:
        return ""
    key = experiment["variable"]
    before, after = cfg["fixed_environment"][key], cfg["channel_shift"]["environment"][key]
    unit = experiment["unit"]
    if key == "horizontal_distance_m":
        before, after, unit = before / 1000, after / 1000, "km"
    return f"{experiment['label']}: {before:g} -> {after:g} {unit}"


def plot_training(output, alpha=0.1, target_ber=1e-5):
    output = Path(output)
    cfg = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    shift = (cfg.get("channel_shift") or {}).get("episode")
    evaluation_alpha = cfg["training"].get("evaluation_ema_alpha", 0.2)
    train = read_csv(output / "training_history.csv")
    action_path = output / "training_actions.csv"
    action_rows = read_csv(action_path) if action_path.exists() else []
    evaluation_rows = read_csv(output / "validation_history.csv")
    validation = [r for r in evaluation_rows if r.get("evaluation_event") != "before_update"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    x, xv = column(train, "episode"), column(validation, "episode")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = [("mean_reward", "Mean reward"), ("loss", "TD loss"),
                      ("mean_tx_power_kw", "Mean TX power (kW)"),
                      ("ber_target_met_fraction_estimate", "Estimated BER target met fraction")]
    for ax, (key, label) in zip(axes.flat, specifications):
        ax.plot(x, column(train, key), color="#86B1CC", alpha=0.25, linewidth=0.6)
        ax.plot(x, phase_ema(train, key, alpha, shift), color="#176B9B", linewidth=1.5,
                label="Training EMA (display only)")
        if key != "loss":
            ax.plot(xv, column(validation, key), "o", alpha=0.4, color="#2E86AB", markersize=2)
            ax.plot(xv, phase_ema(validation, key, evaluation_alpha, shift), color="#2E86AB",
                    linewidth=1.7, label="Fixed evaluation EMA")
        if shift is not None:
            ax.axvline(shift + 0.5, color="#666666", linestyle=":", linewidth=1)
        ax.set(xlabel="Training episode", ylabel=label)
        ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Tensor DQN training and fixed-scenario evaluation")
    fig.savefig(output / "training_convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    training_smooth = plot_learning_ber(axes[0], train, alpha, shift, target_ber)
    axes[0].set_title("DQN learning: exploration is allowed to exceed the final BER target")
    plot_power_comparison(axes[1], train, alpha, shift, action_rows)
    axes[1].set(xlabel="Training episode", ylabel="TX power (kW)")
    axes[1].grid(alpha=0.18)
    if shift is not None:
        axes[1].axvline(shift + 0.5, color="#666666", linestyle=":")
    fig.savefig(output / "ber_vs_episode.png", dpi=180)
    plt.close(fig)
    with (output / "training_smoothed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "training_ber_raw", "training_ber_ema_display_only"])
        writer.writerows((r["episode"], r.get("ber_estimate"), value) for r, value in zip(train, training_smooth))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    smooth = plot_policy_ber(axes[0], validation, evaluation_alpha, shift, target_ber)
    jump = [r for r in evaluation_rows if r.get("evaluation_event") == "before_update"]
    if jump:
        axes[0].scatter(column(jump, "episode"), ber_plot_values(jump)[0], color="#222222", marker="x",
                        label="Old raw configuration on new channel (diagnostic)")
    axes[0].axhline(target_ber, color="#AD3434", linestyle="--", label=f"BER target {target_ber:g}")
    set_ber_limits(axes[0], validation + jump, target_ber)
    axes[0].set(ylabel="Payload BER estimate", title="Raw DQN evaluation: no exploration, no BER prefilter")
    axes[0].legend(fontsize=8)
    axes[1].step(xv, column(validation, "mean_tx_power_kw"), where="post", color="#176B9B")
    axes[1].set(xlabel="Training episode", ylabel="Evaluation power (kW)")
    for ax in axes:
        ax.grid(alpha=0.18)
        if shift is not None:
            ax.axvline(shift + 0.5, color="#666666", linestyle=":")
    fig.savefig(output / "policy_evaluation.png", dpi=180)
    plt.close(fig)
    deployment_path = output / "deployment_history.csv"
    if deployment_path.exists():
        deployed = read_csv(deployment_path)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
        plot_policy_ber(axes[0], deployed, evaluation_alpha, shift, target_ber)
        axes[0].axhline(target_ber, color="#AD3434", linestyle="--", label=f"BER target {target_ber:g}")
        set_ber_limits(axes[0], deployed, target_ber)
        axes[0].set(ylabel="Payload BER estimate", title="Deployment: BER filtering and switching control enabled")
        axes[0].legend(fontsize=8)
        axes[1].step(column(deployed, "episode"), column(deployed, "tx_power_kw"), where="post", color="#176B9B")
        axes[1].set(xlabel="Training episode", ylabel="Executed power (kW)")
        for ax in axes:
            ax.grid(alpha=0.18)
            if shift is not None:
                ax.axvline(shift + 0.5, color="#666666", linestyle=":")
        fig.savefig(output / "deployment_performance.png", dpi=180)
        plt.close(fig)
    with (output / "validation_smoothed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "ber_estimate_raw", "ber_ema_display_only", "phase"])
        writer.writerows((r["episode"], r.get("ber_estimate"), value,
                          "post" if shift is not None and float(r["episode"]) > shift else "pre")
                         for r, value in zip(validation, smooth))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)
    plot_ber_series(axes[0], train, alpha=0.2, linewidth=0.5, label="Training BER with exploration and drift")
    train_smooth = phase_ema(train, "ber_estimate", alpha, shift)
    axes[0].semilogy(x, np.where(train_smooth > 0, train_smooth, np.nan), color="#176B9B", label="Training EMA")
    plot_ber_series(axes[1], train, key="greedy_ber_estimate", resolution_key="greedy_ber_resolution_limit",
                    linewidth=0.6, alpha=0.65, label="Direct model choice, no exploration")
    for ax in axes:
        ax.axhline(target_ber, color="#AD3434", linestyle="--")
        if shift is not None:
            ax.axvline(shift + 0.5, color="#666666", linestyle=":")
        ax.grid(which="major", alpha=0.18)
        ax.set_ylabel("Payload BER estimate")
        ax.legend(fontsize=8)
    axes[1].set_xlabel("Training episode")
    fig.suptitle("Diagnostic curves; zero-count markers are display limits")
    fig.savefig(output / "training_ber_diagnostics.png", dpi=180)
    plt.close(fig)
    plot_action_iterations(output)


def plot_action_iterations(output):
    output = Path(output)
    rows = read_csv(output / "training_history.csv")
    if not rows or "greedy_frequency_hz" not in rows[0]:
        return
    cfg = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    shift = (cfg.get("channel_shift") or {}).get("episode")
    modes = [("frequency_hz", "Selected frequency (Hz)", None),
             ("waveform_mode", "Waveform mode index", None),
             ("modulation", "Selected modulation", ["BPSK", "QPSK", "MSK"]),
             ("symbol_rate", "Symbol rate (sym/s)", None),
             ("coding_scheme", "Selected coding scheme", ["Polar", "CC", "BCH"]),
             ("code_rate", "Nominal code rate", None),
             ("payload_rate", "Payload code rate", None),
             ("tx_power_kw", "TX power (kW)", None)]
    prefixes = ["deployed_", "greedy_"] if "deployed_frequency_hz" in rows[0] else ["greedy_"]
    for prefix in prefixes:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
        for ax, (key, label, categories) in zip(axes.flat, modes):
            if categories:
                values = np.array([categories.index(r[prefix + key]) if r.get(prefix + key) in categories else np.nan for r in rows])
                ax.set_yticks(range(len(categories)), categories)
            else:
                values = column(rows, prefix + key)
            ax.step(column(rows, "episode"), values, where="post", linewidth=1.2, color="#176B9B")
            if shift is not None:
                ax.axvline(shift + 0.5, color="#666666", linestyle=":", linewidth=1)
            ax.set(xlabel="Training episode", ylabel=label)
            if key == "tx_power_kw":
                ax.set_ylim(0, max(cfg["actions"]["tx_power_kw"]) * 1.1)
                ax.set_yticks(power_ticks(cfg["actions"]["tx_power_kw"], 6))
                ax.tick_params(axis="y", labelsize=7)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.grid(alpha=0.18)
        guarded = prefix == "deployed_"
        heading = "Executed configurations with switching control" if guarded else "Direct DQN choices without switching control"
        scenario = experiment_label(cfg)
        fig.suptitle((scenario + "\n" if scenario else "") + heading)
        filename = "deployment_decision_variables_vs_episode.png" if guarded else "greedy_decision_variables_vs_episode.png"
        fig.savefig(output / filename, dpi=180)
        if not guarded:
            fig.savefig(output / "raw_decision_variables_vs_episode.png", dpi=180)
            if not (output / "training_actions.csv").exists():
                fig.savefig(output / "decision_variables_vs_episode.png", dpi=180)
        plt.close(fig)
    action_path = output / "training_actions.csv"
    if action_path.exists():
        plot_sampled_action_iterations(output, rows, read_csv(action_path), cfg)


def phase_rolling_ratio(numerator, denominator, window, shift_episode=None):
    result = np.full(len(numerator), np.nan)
    boundaries = [0, len(numerator)]
    if shift_episode is not None and 0 < int(shift_episode) < len(numerator):
        boundaries.insert(1, int(shift_episode))
    for start, end in zip(boundaries, boundaries[1:]):
        top = np.r_[0.0, np.cumsum(numerator[start:end])]
        bottom = np.r_[0.0, np.cumsum(denominator[start:end])]
        right = np.arange(1, end - start + 1)
        left = np.maximum(0, right - window)
        weights = bottom[right] - bottom[left]
        np.divide(top[right] - top[left], weights, out=result[start:end], where=weights > 0)
    return result


def plot_sampled_action_iterations(output, history, actions, cfg):
    if not actions:
        return
    shift = (cfg.get("channel_shift") or {}).get("episode")
    window = cfg["training"].get("decision_plot_window", 25)
    alpha = cfg["training"]["ema_alpha"]
    count = int(max(column(history, "episode").max(),
                    column(actions, "episode").max()))
    frame_episode = column(actions, "episode").astype(int) - 1
    positions = column(actions, "episode_position")
    episode_x = np.arange(1, count + 1)
    panels = [("frequency_hz", "Selected frequency (Hz)", None),
              ("waveform_mode", "Waveform mode index", list(range(len(cfg["actions"]["waveform_mode"])))),
              ("modulation", "Modulation", ["BPSK", "QPSK", "MSK"]),
              ("symbol_rate", "Symbol rate (sym/s)", None),
              ("coding_scheme", "Coding scheme", ["Polar", "CC", "BCH"]),
              ("code_rate", "Nominal code rate", None),
              ("payload_rate", "Payload code rate", None),
              ("tx_power_kw", "TX power (kW)", None)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for ax, (key, label, categories) in zip(axes.flat, panels):
        if categories is None:
            values = column(actions, key)
            valid = np.isfinite(values)
            totals = np.bincount(frame_episode[valid], weights=values[valid], minlength=count)
            numbers = np.bincount(frame_episode[valid], minlength=count)
            means = np.divide(totals, numbers, out=np.full(count, np.nan), where=numbers > 0)
            mean_rows = [{"episode": int(e), "value": float(v)} for e, v in zip(episode_x, means)]
            ax.scatter(positions[valid], values[valid], s=3, alpha=0.13, color="#76A9CE",
                       rasterized=True, label="Actual training actions")
            ax.plot(episode_x, phase_ema(mean_rows, "value", alpha, shift), color="#176B9B",
                    linewidth=1.6, label="Episode mean EMA (display only)")
            ax.legend(fontsize=6.5, loc="best")
            if key == "tx_power_kw":
                ax.set_ylim(0, max(cfg["actions"]["tx_power_kw"]) * 1.1)
                ax.set_yticks(power_ticks(cfg["actions"]["tx_power_kw"], 6))
                ax.tick_params(axis="y", labelsize=7)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            if key == "waveform_mode":
                values = column(actions, key)
            else:
                values = np.array([categories.index(row[key]) if row.get(key) in categories else np.nan for row in actions])
            valid = np.isfinite(values)
            denominator = np.bincount(frame_episode[valid], minlength=count)
            ratios = []
            for category in range(len(categories)):
                numerator = np.bincount(frame_episode[valid & (values == category)], minlength=count)
                ratios.append(phase_rolling_ratio(numerator, denominator, window, shift))
            image = ax.imshow(np.asarray(ratios), aspect="auto", origin="lower", interpolation="nearest",
                              extent=(0.5, count + 0.5, -0.5, len(categories) - 0.5),
                              vmin=0, vmax=1, cmap="Blues")
            ax.set_yticks(range(len(categories)), categories)
            ax.set_title(f"Trailing {window}-episode selection proportions", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02).set_label("Proportion", fontsize=7)
        if shift is not None:
            ax.axvline(shift + 0.5, color="#666666", linestyle=":", linewidth=1)
        ax.set(xlabel="Training episode", ylabel=label, xlim=(0.5, count + 0.5))
        ax.grid(alpha=0.15)
    scenario = experiment_label(cfg)
    fig.suptitle((scenario + "\n" if scenario else "")
                 + "Actual training choices\n"
                 + "Raw actions include exploration; smoothing and category proportions are display only",
                 fontsize=12)
    fig.savefig(output / "decision_variables_vs_episode.png", dpi=180)
    fig.savefig(output / "training_decision_variables_vs_episode.png", dpi=180)
    plt.close(fig)


def plot_table(table, output):
    schemes = list(dict.fromkeys(scheme for scheme, _ in table.cfg["actions"]["coding_mode"]))
    modulations = table.cfg["actions"]["modulation"]
    fig, axes = plt.subplots(len(schemes), len(modulations),
                             figsize=(15, 4.5 * len(schemes)), constrained_layout=True,
                             squeeze=False)
    for row, scheme in enumerate(schemes):
        rates = [rate for s, rate in table.cfg["actions"]["coding_mode"] if s == scheme]
        for col, modulation in enumerate(modulations):
            ax = axes[row, col]
            for rate in rates:
                ber, _ = table.curves[(scheme, modulation, rate)]
                ax.semilogy(table.curve_knots[(scheme, modulation, rate)], ber, "o-", markersize=3, label=f"Nominal R={rate:.3f}")
            ax.set(xlabel="Es/N0 (dB)", ylabel="Smoothed payload BER estimate",
                   title=f"{scheme} / {modulation}")
            ax.grid(which="both", alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Real Monte Carlo BER tables by coding scheme and modulation")
    fig.savefig(Path(output) / "ber_table.png", dpi=180)
    plt.close(fig)

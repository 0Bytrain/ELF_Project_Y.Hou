from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

try:  # Optional: keep the training loop usable without tqdm installed.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    tqdm = None

from .antenna import AntennaModel
from .config import (ENVIRONMENT_KEYS, action_shape, coding_mode_index, coding_modes,
                     fingerprint, resolve_path, save_json, waveform_mode_index, waveform_modes)
from .dqn import DQNAgent
from .environment import CommunicationEnvironment
from .performance import table_specification, utility
from .convergence import tail_statistics, policy_tail_statistics, candidate_lock_statistics
from .selection import StableDecisionController, learning_rate_at
from .range_model import RangeModel


def summarize(records):
    transmitted = [row for row in records if row["transmitted"]]
    bits = sum(row["payload_bits"] for row in transmitted)
    mean_frame = float(np.mean([row["frame_duration_s"] for row in transmitted])) if transmitted else 1.0
    seconds = sum(row.get("frame_duration_s", mean_frame) for row in records)
    powers = [row["tx_power_kw"] for row in transmitted]
    esn0 = [float(row["esn0_db"]) for row in transmitted if row.get("esn0_db") is not None]
    snr = [float(row["snr_db"]) for row in transmitted if row.get("snr_db") is not None]
    return {
        "mean_reward": float(np.mean([row["reward"] for row in records])),
        "ber_estimate": float(sum(row["ber_estimate"] * row["payload_bits"] for row in transmitted) / bits) if bits else None,
        "ber_resolution_limited_fraction": float(np.mean([row.get("ber_resolution_limited", False) for row in transmitted])) if transmitted else None,
        "ber_resolution_limit": max((row.get("ber_resolution_limit", 0.0) for row in transmitted), default=None),
        "goodput_bps_estimate": float(sum(row["goodput_bps_estimate"] * row["frame_duration_s"] for row in transmitted) / seconds),
        # Fraction of transmitted frames, not all records: an outage is not a
        # frame that failed the BER target and must not dilute this metric.
        "ber_target_met_fraction_estimate": float(sum(row["meets_ber_target_estimate"] for row in transmitted) / len(transmitted)) if transmitted else None,
        "outage_fraction": float(1 - len(transmitted) / len(records)),
        "mean_tx_power_kw": float(np.mean(powers)) if powers else None,
        "power_switch_fraction": float(sum(bool(row.get("power_changed")) for row in transmitted) / len(transmitted)) if transmitted else None,
        # BER is only meaningful together with its operating SNR.  These are
        # simple frame means over the transmitted records; validation uses a
        # fixed scenario, while training includes channel drift and occupancy.
        "mean_esn0_db": float(np.mean(esn0)) if esn0 else None,
        "median_esn0_db": float(np.median(esn0)) if esn0 else None,
        "min_esn0_db": float(np.min(esn0)) if esn0 else None,
        "mean_snr_db": float(np.mean(snr)) if snr else None,
        "configuration_switch_fraction": float(np.mean([row.get("switched", False) for row in transmitted])) if transmitted else None,
        "power_optimality_gap_kw": float(np.mean([row["power_optimality_gap_kw"] for row in transmitted])) if transmitted and all(r.get("power_optimality_gap_kw") is not None for r in transmitted) else None,
        "minimum_feasible_tx_power_kw": min((row["minimum_feasible_tx_power_kw"] for row in transmitted
                                            if row.get("minimum_feasible_tx_power_kw") is not None), default=None),
        "ber_baseline_used": next((row.get("ber_baseline_used") for row in records if row.get("ber_baseline_used") is not None), None),
    }


def convergence_stats(rows, window=10, tolerance=0.15, key="ber_estimate"):
    return tail_statistics(rows, window, tolerance, key)


def phase_epsilon(settings, episode, phase_start, phase_end, start_epsilon):
    length = phase_end - phase_start + 1
    fraction = min(1.0, (episode - phase_start) / max(1.0, length * settings["epsilon_decay_fraction"]))
    if settings.get("exploration_schedule", "linear") == "phase_cosine":
        fraction = 0.5 * (1 - np.cos(np.pi * fraction))
    return float(start_epsilon + fraction * (settings["epsilon_end"] - start_epsilon))


def evaluate_policy(agent, cfg, table, episode_count=None, occupied_bands=None,
                    environment=None, dynamic=False, ber_baseline=None,
                    power_cap_kw=None, ber_filter=False):
    """Evaluate the greedy policy under a controlled, fixed scenario.

    Training intentionally uses random occupancy and slow environment drift.
    Validation must not add that noise again, otherwise the reported BER mixes
    policy changes with random band occupancy.  Callers that explicitly want a
    robustness evaluation can pass ``dynamic=True``.

    The absolute BER objective is identical before and after a channel change.
    ``ber_baseline`` remains an accepted legacy argument but does not change
    the reward threshold.
    """
    records = []
    count = episode_count or cfg["training"]["evaluation_episodes"]
    if environment is None:
        # A validation/probe call must not sample a new physical environment
        # per episode.  The reference environment is the controlled fallback;
        # dynamic=True remains available for explicitly requested robustness
        # evaluations.
        environment = cfg.get("link", {}).get("reference_environment")
    for episode in range(count):
        env = CommunicationEnvironment(cfg, table, seed=cfg["seed"] + 100000 + episode, ber_filter=ber_filter)
        env.ber_baseline = ber_baseline
        if hasattr(env, "set_power_cap"):
            env.set_power_cap(power_cap_kw)
        state, mask = env.reset(environment=environment, occupied_bands=occupied_bands,
                                dynamic=dynamic)
        if hasattr(env, "estimated_feasible_mask"):
            env.estimated_feasible_mask()
        done = False
        while not done:
            action = agent.act(state, mask, epsilon=0.0)
            state, _, done, mask, info = env.step(action)
            info["evaluation_episode"] = episode + 1
            records.append(info)
    metrics = summarize(records)
    metrics["policy_signature"] = fingerprint([r["action_index"] for r in records]) if all("action_index" in r for r in records) else None
    metrics["initial_action_index"] = records[0].get("action_index") if records else None
    return metrics, records


def checkpoint_contract(cfg):
    link = cfg["link"]
    nf = len(cfg["actions"]["frequency_hz"])
    nwave = len(waveform_modes(cfg))
    state_dim = 6 + 3 * nf + nf * nwave + nf * nwave + 5
    contract = {
        "state_schema": 4,
        "selection_schema": "raw_training_guarded_deployment_v2",
        "training_schema": "observed_actions_phase_exploration_balanced_replay_v3",
        "tensor_additive_value_heads": cfg["training"].get("additive_value_heads", False),
        "training_ber_filter": cfg["training"].get("ber_feasibility_filter", False),
        "decision": cfg.get("decision", {}),
        "ber_estimator": "empirical_monotone_linear_v2",
        "reward_schema": "fixed_absolute_ber_v2",
        "state_dim": state_dim,
        "actions": cfg["actions"],
        "waveform_modes": [list(p) for p in waveform_modes(cfg)],
        "coding_modes": [list(p) for p in coding_modes(cfg)],
        "polar": cfg["polar"],
        "normalization_ranges": cfg["environment"]["ranges"],
        "objective": cfg["objective"],
        # Channel, antenna, noise, guard-band and occupancy changes alter the
        # state/mask/reward semantics even when the action grid is unchanged.
        "link_contract": fingerprint(cfg["link"]),
        "environment_contract": fingerprint(cfg["environment"]),
        "fixed_environment": cfg.get("fixed_environment"),
        "channel_shift": cfg.get("channel_shift"),
        "link_calibration": {
            "field_anchor_distance_m": link["field_anchor_distance_m"],
            "field_anchor_e_db_uv_m": link["field_anchor_e_db_uv_m"],
            "reference_environment": link["reference_environment"],
            "reference_signal_rms_pt": link["reference_signal_rms_pt"],
            "reference_tx_power_kw": link["reference_tx_power_kw"],
            "propagation": {
                "model": link["propagation_model"],
                "mode": link.get("ionosphere_mode"),
                "beta_scale": link.get("beta_scale"),
                "min_km": link.get("propagation_min_km"),
                "params": link.get("ionosphere"),
                "table_hash": RangeModel(link).fingerprint() if link["propagation_model"] == "lwpc_table" else None,
            },
        },
        "ber_table_signature": fingerprint(table_specification(cfg)),
    }
    if link.get("source_model", "anchor") == "current_moment":
        # Keep anchor-mode contracts compatible with existing checkpoints; the
        # optional source model gets its own explicit contract entries.
        contract["link_calibration"]["source_model"] = "current_moment"
        contract["link_calibration"]["antenna"] = AntennaModel(link).fingerprint()
    return contract


def save_checkpoint(agent, path, cfg, episode, metric):
    agent.save(path)
    save_json(str(path) + ".json", {
        "contract": checkpoint_contract(cfg), "contract_hash": fingerprint(checkpoint_contract(cfg)),
        "episode": episode, "validation": metric, "tensor_rank": cfg["training"]["tensor_rank"],
        "hidden_size": cfg["training"]["hidden_size"], "run_label": cfg.get("run_label", "relative_link_simulation"),
    })


def load_agent(cfg, table, checkpoint=None):
    checkpoint = resolve_path(checkpoint or str(Path(cfg["output_dir"]) / "best_model.pt"))
    metadata = json.loads(Path(str(checkpoint) + ".json").read_text(encoding="utf-8"))
    if metadata["contract_hash"] != fingerprint(checkpoint_contract(cfg)):
        raise ValueError("Checkpoint action/codec/state normalization mismatch; use its run_config.json")
    env = CommunicationEnvironment(cfg, table)
    settings = dict(cfg["training"], tensor_rank=metadata["tensor_rank"], hidden_size=metadata["hidden_size"])
    agent = DQNAgent(env.state_dim, action_shape(cfg), settings, cfg["seed"], device=settings["device"])
    agent.load(checkpoint)
    return agent


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def action_snapshot(env, index, prefix):
    keys = ("frequency_hz", "tx_power_kw", "coding_scheme", "code_rate", "payload_rate",
            "modulation", "symbol_rate", "ber_estimate", "ber_resolution_limited",
            "ber_resolution_limit", "esn0_db")
    values = env.action_metrics(index) if index >= 0 else {}
    result = {prefix + "action_index": index}
    result.update({prefix + key: values.get(key) for key in keys})
    result[prefix + "waveform_mode"] = waveform_mode_index(env.actions[index], env.cfg) if index >= 0 else None
    result[prefix + "coding_mode"] = coding_mode_index(env.actions[index], env.cfg) if index >= 0 else None
    return result


def train(cfg, table, scenario=None, occupied_bands=None):
    settings = cfg["training"]
    torch.set_num_threads(settings.get("torch_threads", 2))
    output = resolve_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_config.json", cfg)
    env = CommunicationEnvironment(cfg, table, seed=cfg["seed"],
                                   ber_filter=settings.get("ber_feasibility_filter", False))
    deployment_env = CommunicationEnvironment(cfg, table, seed=cfg["seed"] + 200000, ber_filter=True)
    agent = DQNAgent(env.state_dim, action_shape(cfg), settings, cfg["seed"], device=settings["device"])
    history, validation_history, deployment_history = [], [], []
    exploration_anchor, anchor_score = None, -np.inf
    controller = StableDecisionController(cfg)
    stability = settings.get("convergence", {})
    frozen = False
    phase_converged = False
    candidate_lock_events = []
    convergence_events = []
    best_reward = -np.inf
    start = time.perf_counter()
    fixed_environment = cfg.get("fixed_environment")
    # run.py supplies a scenario environment; without a fixed_environment it
    # must drive training, not only the final decision.  Validation still uses
    # a controlled reference/scenario channel below.
    training_environment = fixed_environment if fixed_environment is not None else scenario
    channel_shift = cfg.get("channel_shift")#信道中途切换配置（可选）
    if channel_shift is not None:
        shifted = channel_shift["environment"]
        if set(shifted) != set(ENVIRONMENT_KEYS) or not all(np.isfinite(v) for v in shifted.values()):
            raise ValueError("channel_shift.environment must give exactly the six finite physical parameters")
        if not 0 < int(channel_shift["episode"]) < settings["episodes"]:
            raise ValueError("channel_shift.episode must lie inside the run")
    if (channel_shift or {}).get("power_staging", {}).get("enabled", False):
        raise ValueError("Disable power_staging: all power levels must remain available for BER-constrained power minimization")
    power_cap = None
    initial_environment = training_environment or cfg["link"]["reference_environment"]
    initial_metrics, _ = evaluate_policy(agent, cfg, table, environment=initial_environment,
                                         occupied_bands=occupied_bands, dynamic=False)
    validation_history.append({"episode": 0, "evaluation_event": "untrained", **initial_metrics})
    progress = None
    if tqdm is not None:
        progress = tqdm(range(1, settings["episodes"] + 1), total=settings["episodes"],
                        desc="training", unit="episode", dynamic_ncols=True,
                        mininterval=1.0, leave=True)
        episode_iter = progress
    else:
        episode_iter = range(1, settings["episodes"] + 1)
    logged_metrics = ("action_index", "frequency_hz", "tx_power_kw", "modulation", "symbol_rate",
                      "coding_scheme", "code_rate", "payload_rate", "ber_estimate",
                      "ber_resolution_limited", "ber_resolution_limit", "esn0_db", "reward",
                      "transmitted", "outage", "meets_ber_target_estimate")
    action_fields = ("episode", "step", "episode_position", "phase_start", "epsilon", "action_source",
                     "waveform_mode", "coding_mode") + logged_metrics
    action_stream = (output / "training_actions.csv").open("w", encoding="utf-8-sig", newline="")
    action_log = csv.DictWriter(action_stream, fieldnames=action_fields)
    action_log.writeheader()
    try:
        for episode in episode_iter:#主循环
            if progress is None and (episode == 1 or episode % 10 == 0):
                print(f"Episode {episode}/{settings['episodes']} elapsed={time.perf_counter() - start:.1f}s",
                      flush=True)
            phase_environment = training_environment
            if channel_shift is not None and episode > int(channel_shift["episode"]):
                phase_environment = channel_shift["environment"]#切换点之后按新信道继续训练与验证
            # Validation must be a controlled channel even when training is
            # allowed to sample the environment randomly.
            validation_environment = (phase_environment if phase_environment is not None
                                      else cfg["link"]["reference_environment"])
            phase_start = int(channel_shift["episode"]) + 1 if channel_shift and episode > int(channel_shift["episode"]) else 1
            phase_end = settings["episodes"] if not channel_shift or phase_start > 1 else int(channel_shift["episode"])
            phase_episode = episode - phase_start + 1
            start_epsilon = (channel_shift.get("epsilon_after", settings["epsilon_end"])
                             if phase_start > 1 else settings["epsilon_start"])
            epsilon = phase_epsilon(settings, episode, phase_start, phase_end, start_epsilon)
            lr = learning_rate_at(settings, phase_episode, phase_end - phase_start + 1)
            for group in agent.optimizer.param_groups:
                group["lr"] = lr
            if channel_shift is not None and episode == int(channel_shift["episode"]) + 1:
                frozen = False
                phase_converged = False
                exploration_anchor, anchor_score = None, -np.inf
                if settings.get("reset_replay_on_channel_shift", False):
                    agent.reset_replay()
                # Reward and BER threshold remain identical in both phases. Select
                # the best post-change deployment checkpoint within its own phase.
                best_reward = -np.inf
                # Measure the old policy on the new channel before any update.  A
                # regular evaluation every 20 episodes otherwise hides the jump.
                old_index = history[-1]["greedy_action_index"] if history else controller.previous_action
                if old_index is not None and old_index >= 0:
                    env.reset(environment=validation_environment, occupied_bands=occupied_bands, dynamic=False)
                    old = env.action_metrics(old_index)
                    old.update(transmitted=bool(old["legal"]), outage=not bool(old["legal"]),
                               reward=0.0, action_index=old_index)
                    jump_metrics = summarize([old])
                    validation_history.append({"episode": episode, "evaluation_event": "before_update",
                                               "diagnostic_only": True, "power_cap_kw": power_cap, **jump_metrics})
            state, mask = env.reset(environment=phase_environment, occupied_bands=occupied_bands,
                                    dynamic=settings.get("dynamic_environment", True))
            if phase_environment is None or settings.get("dynamic_environment", True):
                exploration_anchor, anchor_score = None, -np.inf
            if frozen:
                epsilon = 0.0
            records, losses = [], []
            done = False
            while not done:
                action = agent.act(state, mask, epsilon, exploration_anchor=exploration_anchor)
                action_source = agent.last_action_source
                next_state, reward, done, next_mask, info = env.step(action)
                if action >= 0 and not frozen:
                    feasible = bool(info.get("meets_ber_target_estimate", False))
                    agent.remember(state, action, reward, next_state, done, next_mask, feasible=feasible)
                    if feasible and phase_environment is not None and not settings.get("dynamic_environment", True):
                        score = utility(info, cfg, switched=False)
                        if score > anchor_score:
                            exploration_anchor, anchor_score = action, score
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(float(loss))
                records.append(info)
                action_log.writerow({"episode": episode, "step": len(records),
                                     "episode_position": episode - 0.5 + (len(records) - 0.5) / settings["steps_per_episode"],
                                     "phase_start": phase_start, "epsilon": epsilon, "action_source": action_source,
                                     "waveform_mode": waveform_mode_index(env.actions[action], cfg) if action >= 0 else None,
                                     "coding_mode": coding_mode_index(env.actions[action], cfg) if action >= 0 else None,
                                     **{key: info.get(key) for key in logged_metrics}})
                state, mask = next_state, next_mask
            env.set_power_cap(power_cap)
            probe_state, probe_mask = env.reset(
                environment=validation_environment, occupied_bands=occupied_bands, dynamic=False)
            greedy_index = int(agent.act(probe_state, probe_mask, epsilon=0.0))
            greedy = action_snapshot(env, greedy_index, "greedy_")
            _, deployment_mask = deployment_env.reset(environment=validation_environment,
                                                       occupied_bands=occupied_bands, dynamic=False)
            deployment_env.previous_action = controller.previous_action
            safe_candidate = greedy_index if greedy_index >= 0 and deployment_mask[greedy_index] else None
            selected, selection_reason = controller.select(deployment_env, agent, deployment_env.state(),
                                                           deployment_mask, candidate=safe_candidate)
            deployed = action_snapshot(deployment_env, selected, "deployed_")
            _, _, _, _, deployment = deployment_env.step(selected)
            deployment["raw_candidate_ber_estimate"] = greedy.get("greedy_ber_estimate")
            deployment["safety_filter_changed_candidate"] = safe_candidate is None and greedy_index >= 0
            deployment_history.append({"episode": episode, "selection_reason": selection_reason,
                                       "greedy_action_index": greedy_index, **deployment})
            row = {"episode": episode, "epsilon": float(epsilon), **summarize(records), **greedy, **deployed,
                   "best_observed_feasible_action_index": exploration_anchor,
                   "selection_reason": selection_reason, "learning_rate": lr, "learning_active": not frozen,
                   "ber_baseline_used": env.ber_baseline, "power_cap_kw": power_cap,
                   "loss": float(np.mean(losses)) if losses else None, "elapsed_s": time.perf_counter() - start}
            history.append(row)
            if (episode == 1 or episode % settings["evaluation_interval"] == 0
                    or episode == settings["episodes"]
                    or (channel_shift and episode in (int(channel_shift["episode"]), int(channel_shift["episode"]) + 1))):
                metrics, evaluation_records = evaluate_policy(
                    agent, cfg, table, occupied_bands=occupied_bands,
                    environment=validation_environment, dynamic=False,
                    ber_baseline=env.ber_baseline, power_cap_kw=power_cap)
                validation_history.append({"episode": episode, "evaluation_event": "after_update", "power_cap_kw": power_cap,
                                           "training_epsilon": epsilon,
                                           **metrics})
                if metrics["mean_reward"] > best_reward:
                    best_reward = metrics["mean_reward"]
                    save_checkpoint(agent, output / "best_model.pt", cfg, episode, metrics)
                    write_csv(output / "best_policy_evaluation.csv", evaluation_records)
                write_csv(output / "training_history.csv", history)
                write_csv(output / "validation_history.csv", validation_history)
                write_csv(output / "deployment_history.csv", deployment_history)
                action_stream.flush()
                phase_rows = [r for r in validation_history if float(r["episode"]) >= phase_start]
                policy_stats = policy_tail_statistics(phase_rows, cfg["objective"]["target_ber"],
                                                      stability.get("window", 20), stability)
                can_lock = (stability.get("enabled", False) and training_environment is not None
                            and not settings.get("dynamic_environment", True)
                            and phase_episode >= stability.get("min_phase_episodes", 200))
                if can_lock and not frozen:
                    candidate_stats = candidate_lock_statistics(phase_rows, cfg["objective"]["target_ber"], stability)
                    if candidate_stats["ready"]:
                        frozen = True
                        candidate_lock_events.append({"episode": episode, "phase_start": phase_start, **candidate_stats})
                        save_checkpoint(agent, output / ("pre_shift_candidate_model.pt" if phase_end < settings["episodes"]
                                                         else "candidate_model.pt"), cfg, episode, metrics)
                        print(f"Objective reached at episode {episode}; model updates paused for fixed-policy confirmation.", flush=True)
                if (stability.get("enabled", False) and training_environment is not None
                        and not phase_converged and policy_stats["converged"]
                        and phase_episode >= stability.get("min_phase_episodes", 200)):
                    convergence_events.append({"episode": episode, "phase_start": phase_start, **policy_stats})
                    save_checkpoint(agent, output / ("pre_shift_stable_model.pt" if phase_end < settings["episodes"]
                                                     else "stable_model.pt"), cfg, episode, metrics)
                    frozen = True
                    phase_converged = True
                    save_checkpoint(agent, output / "best_model.pt", cfg, episode, metrics)
                    write_csv(output / "best_policy_evaluation.csv", evaluation_records)
                    print(f"Policy stable at episode {episode}; estimated BER target and minimum model power verified.", flush=True)
                    if phase_end == settings["episodes"] and stability.get("stop_after_final_convergence", True):
                        break
                print(f"Episode {episode}/{settings['episodes']} reward={row['mean_reward']:.3f} "
                      f"validation={metrics['mean_reward']:.3f} BER_est={metrics['ber_estimate']} "
                      f"mean_EsN0={metrics.get('mean_esn0_db')} dB "
                      f"goodput={metrics['goodput_bps_estimate']:.3f} bit/s", flush=True)
    finally:
        action_stream.close()
    if progress is not None:
        progress.close()
    save_checkpoint(agent, output / "last_model.pt", cfg, len(history), validation_history[-1])
    save_json(output / "training_summary.json", {
        "episodes": len(history), "maximum_episodes": settings["episodes"],
        "convergence_events": convergence_events,
        "candidate_lock_events": candidate_lock_events,
        "final_policy_converged": bool(policy_stats["converged"]),
        "updates_stopped_for_convergence": phase_converged,
        "training_ber_filter": settings.get("ber_feasibility_filter", False),
        "controlled_experiment": cfg.get("controlled_experiment"),
        "dynamic_training_environment": settings.get("dynamic_environment", True),
        "early_stopped": len(history) < settings["episodes"],
        "selection": "Raw DQN learning and evaluation; BER feasibility filtering and hysteresis only for deployment",
        "action_shape": list(action_shape(cfg)), "state_dim": env.state_dim,
        "best_validation_reward": best_reward, "elapsed_s": time.perf_counter() - start,
        "training_ber_source": "empirical_monotone_linear_table_with_resolution_flags_not_new_bit_trials",
        "objective_mode": cfg["objective"].get("mode", "weighted"),
        "target_ber": cfg["objective"]["target_ber"],
        "run_label": cfg.get("run_label", "relative_link_simulation"),
        "reliability_certified": False,
        "final_policy_assessment": policy_stats,
        "final_policy_ber_resolved": policy_stats.get("resolved", False),
        "final_estimated_minimum_power_reached": policy_stats.get("minimum_power_verified_in_model", False),
        "exploration_schedule": settings.get("exploration_schedule", "linear"),
        "replay_balance_fraction": settings.get("replay_balance_fraction", 0.0),
        "training_action_file": "training_actions.csv",
        "training_action_source": "all actual per-frame sampled actions, including exploration; no synthetic variation",
    })
    from .plots import plot_training
    plot_training(output, cfg["training"]["ema_alpha"], cfg["objective"]["target_ber"])
    return output

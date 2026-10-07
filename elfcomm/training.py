from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import action_shape, fingerprint, resolve_path, save_json
from .dqn import DQNAgent
from .environment import CommunicationEnvironment


def summarize(records):
    transmitted = [row for row in records if row["transmitted"]]
    bits = sum(row["payload_bits"] for row in transmitted)
    seconds = sum(row.get("frame_duration_s", 1.0) for row in records)
    return {
        "mean_reward": float(np.mean([row["reward"] for row in records])),
        "ber_estimate": float(sum(row["ber_estimate"] * row["payload_bits"] for row in transmitted) / bits) if bits else None,
        "goodput_bps_estimate": float(sum(row["goodput_bps_estimate"] * row["frame_duration_s"] for row in transmitted) / seconds),
        "ber_target_met_fraction_estimate": float(sum(row["meets_ber_target_estimate"] for row in transmitted) / len(records)),
        "outage_fraction": float(1 - len(transmitted) / len(records)),
    }


def evaluate_policy(agent, cfg, table, episode_count=None, occupied_bands=None, environment=None):
    records = []
    count = episode_count or cfg["training"]["evaluation_episodes"]
    for episode in range(count):
        env = CommunicationEnvironment(cfg, table, seed=cfg["seed"] + 100000 + episode)
        state, mask = env.reset(environment=environment, occupied_bands=occupied_bands)
        done = False
        while not done:
            action = agent.act(state, mask, epsilon=0.0)
            state, _, done, mask, info = env.step(action)
            info["evaluation_episode"] = episode + 1
            records.append(info)
    return summarize(records), records


def checkpoint_contract(cfg):
    return {"state_schema": 1, "actions": cfg["actions"], "polar": cfg["polar"],
            "normalization_ranges": cfg["environment"]["ranges"],
            "objective": cfg["objective"]}


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


def train(cfg, table, scenario=None, occupied_bands=None):
    settings = cfg["training"]
    torch.set_num_threads(settings.get("torch_threads", 2))
    output = resolve_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_config.json", cfg)
    env = CommunicationEnvironment(cfg, table, seed=cfg["seed"])   #创建环境
    agent = DQNAgent(env.state_dim, action_shape(cfg), settings, cfg["seed"], device=settings["device"])
    history, validation_history = [], []
    best_reward = -np.inf
    start = time.perf_counter()
    shift = cfg.get("channel_shift") or {}
    shift_enabled = bool(shift.get("enabled"))
    shift_episode = int(shift.get("episode", 0)) if shift_enabled else None
    fixed_environment = cfg.get("fixed_environment")
    for episode in range(1, settings["episodes"] + 1):#主循环
        fraction = min(1, (episode - 1) / max(1, settings["episodes"] * settings["epsilon_decay_fraction"]))
        epsilon = settings["epsilon_start"] + fraction * (settings["epsilon_end"] - settings["epsilon_start"])
        phase_environment = fixed_environment
        if shift_enabled:
            phase_environment = shift["after_environment"] if episode >= shift_episode else shift["before_environment"]
        state, mask = env.reset(environment=phase_environment, occupied_bands=occupied_bands)#每轮的重采样环境（对比实验时固定信道）
        records, losses = [], []
        done = False
        while not done:
            action = agent.act(state, mask, epsilon)
            next_state, reward, done, next_mask, info = env.step(action)
            if action >= 0:
                agent.remember(state, action, reward, next_state, done, next_mask)
                loss = agent.learn()
                if loss is not None:
                    losses.append(float(loss))
            records.append(info)
            state, mask = next_state, next_mask
        probe_state, probe_mask = env.reset(
            environment=phase_environment if phase_environment is not None else
            (scenario if scenario is not None else cfg["link"]["reference_environment"]),
            occupied_bands=occupied_bands, dynamic=False)
        greedy_index = int(agent.act(probe_state, probe_mask, epsilon=0.0))
        greedy = {"greedy_action_index": greedy_index}
        if greedy_index >= 0:
            greedy_action = env.actions[greedy_index]
            greedy.update(greedy_frequency_hz=greedy_action.frequency_hz,
                          greedy_modulation=greedy_action.modulation,
                          greedy_code_rate=greedy_action.code_rate,
                          greedy_symbol_rate=greedy_action.symbol_rate)
        else:
            greedy.update(greedy_frequency_hz=None, greedy_modulation=None,
                          greedy_code_rate=None, greedy_symbol_rate=None)
        row = {"episode": episode, "epsilon": float(epsilon), **summarize(records), **greedy,
               "loss": float(np.mean(losses)) if losses else None, "elapsed_s": time.perf_counter() - start}
        history.append(row)
        if episode == 1 or episode % settings["evaluation_interval"] == 0 or episode == settings["episodes"]:
            metrics, evaluation_records = evaluate_policy(agent, cfg, table, occupied_bands=occupied_bands, environment=phase_environment)
            validation_history.append({"episode": episode, **metrics})
            if metrics["mean_reward"] > best_reward:
                best_reward = metrics["mean_reward"]
                save_checkpoint(agent, output / "best_model.pt", cfg, episode, metrics)
                write_csv(output / "best_policy_evaluation.csv", evaluation_records)
            write_csv(output / "training_history.csv", history)
            write_csv(output / "validation_history.csv", validation_history)
            print(f"Episode {episode}/{settings['episodes']} reward={row['mean_reward']:.3f} "
                  f"validation={metrics['mean_reward']:.3f} BER_est={metrics['ber_estimate']} "
                  f"goodput={metrics['goodput_bps_estimate']:.3f} bit/s", flush=True)
    save_checkpoint(agent, output / "last_model.pt", cfg, settings["episodes"], validation_history[-1])
    save_json(output / "training_summary.json", {
        "episodes": settings["episodes"], "action_shape": list(action_shape(cfg)), "state_dim": env.state_dim,
        "best_validation_reward": best_reward, "elapsed_s": time.perf_counter() - start,
        "training_ber_source": "smoothed_interpolated_monte_carlo_table_not_new_bit_trials",
        "run_label": cfg.get("run_label", "relative_link_simulation"),
        "reliability_certified": False,
    })
    from .plots import plot_channel_shift_recovery, plot_training
    plot_training(output, cfg["training"]["ema_alpha"], cfg["objective"]["target_ber"])
    if shift_enabled:
        plot_channel_shift_recovery(output, shift_episode, cfg)
    return output

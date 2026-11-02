import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from elfcomm.config import load_config
from elfcomm.convergence import policy_tail_statistics, tail_statistics
from elfcomm.environment import CommunicationEnvironment
from elfcomm.plots import phase_ema
from elfcomm.selection import StableDecisionController, learning_rate_at
from tests.test_ber015_revision import make_table
from tests.test_environment_performance import FakeTable


def test_ber_mask_rejects_all_transmissions_when_model_target_unreachable():
    cfg = load_config()
    env = CommunicationEnvironment(cfg, FakeTable(ber=0.2))
    _, mask = env.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    assert env.spectral_mask.any() and not mask.any()
    with pytest.raises(ValueError):
        env.step(0)
    _, _, _, _, info = env.step(-1)
    assert info["outage_reason"] == "ber_target_unreachable"
    assert info["ber_estimate"] is None


def test_safe_mask_and_model_minimum_power_are_reported():
    cfg = load_config()
    env = CommunicationEnvironment(cfg, FakeTable(ber=1e-6))
    _, mask = env.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    index = int(np.flatnonzero(mask)[0])
    _, _, _, _, info = env.step(index)
    assert info["minimum_feasible_tx_power_kw"] == 62.5
    assert info["power_optimality_gap_kw"] == info["tx_power_kw"] - 62.5


def test_batch_ber_filter_uses_same_estimates_as_scalar_lookup(tmp_path):
    table = make_table(tmp_path, [(20, 100), (0, 200)])
    snr = np.array([-10, 0, 0.5, 1, 2, 3])
    values = table.lookup_ber_many("Polar", "BPSK", 0.5, snr)
    np.testing.assert_allclose(values, [table.lookup("Polar", "BPSK", 0.5, s)["ber_estimate"] for s in snr])


class CandidateAgent:
    def __init__(self, index):
        self.index = index

    def act(self, state, mask, epsilon):
        return self.index


def controller_env():
    return SimpleNamespace(actions=[SimpleNamespace(tx_power_kw=p) for p in (125, 250, 62.5, 125)],
                           action_metrics=lambda i: {"goodput_bps_estimate": [0.1, 0.8, 0.05, 0.105][i]})


def test_controller_keeps_feasible_power_and_near_equivalent_configuration():
    control = StableDecisionController({})
    env, mask = controller_env(), np.ones(4, dtype=bool)
    assert control.select(env, CandidateAgent(0), None, mask)[0] == 0
    assert control.select(env, CandidateAgent(1), None, mask)[0] == 0
    assert control.select(env, CandidateAgent(3), None, mask)[0] == 0
    assert control.select(env, CandidateAgent(2), None, mask)[0] == 2


def test_controller_replaces_infeasible_previous_without_waiting():
    control = StableDecisionController({})
    control.previous_action = 2
    env, mask = controller_env(), np.array([True, True, False, True])
    assert control.select(env, CandidateAgent(1), None, mask)[0] == 1
    assert control.select(env, CandidateAgent(-1), None, np.zeros(4, dtype=bool))[0] == -1
    assert control.previous_action is None


def policy_rows():
    return [{"episode": i + 1, "evaluation_event": "after_update", "ber_estimate": 0.05,
             "ber_target_met_fraction_estimate": 1.0, "outage_fraction": 0.0,
             "mean_tx_power_kw": 62.5, "mean_reward": 1.0,
             "power_optimality_gap_kw": 0.0, "policy_signature": "fixed"} for i in range(20)]


def test_policy_stopping_requires_unfiltered_metrics_and_all_frames_meeting_target():
    rows = policy_rows()
    assert policy_tail_statistics(rows, 0.15)["converged"]
    rows[-1]["ber_target_met_fraction_estimate"] = 0.5
    assert not policy_tail_statistics(rows, 0.15)["converged"]
    assert not tail_statistics(rows, target_ber=0.15)["target_met"]


def test_stable_policy_can_be_identified_without_claiming_true_zero_ber():
    rows = policy_rows()
    for row in rows:
        row.update(ber_estimate=0, ber_resolution_limited_fraction=1.0)
    stats = policy_tail_statistics(rows, 0.15)
    assert stats["converged"] and not stats["resolved"]


@pytest.mark.parametrize("key,value", [("power_optimality_gap_kw", 62.5),
                                      ("outage_fraction", 0.1),
                                      ("policy_signature", "different"),
                                      ("mean_reward", 0.5)])
def test_bad_last_window_cannot_be_hidden_by_smoothing_or_old_plateau(key, value):
    rows = policy_rows()
    rows[-1][key] = value
    assert not policy_tail_statistics(rows, 0.15)["converged"]


def test_smoothing_resets_at_channel_change_and_leaves_raw_rows_unchanged():
    rows = [{"episode": 1, "ber_estimate": 0.1}, {"episode": 2, "ber_estimate": 0.0},
            {"episode": 3, "ber_estimate": 0.4}, {"episode": 4, "ber_estimate": 0.2}]
    original = copy.deepcopy(rows)
    np.testing.assert_allclose(phase_ema(rows, "ber_estimate", 0.5, 2), [0.1, 0.05, 0.4, 0.3])
    assert rows == original


def test_learning_rate_restarts_for_each_phase_and_decays_to_final_value():
    settings = {"learning_rate": 1e-4, "learning_rate_final": 3e-5}
    values = [learning_rate_at(settings, i, 100) for i in range(1, 101)]
    assert values[0] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(3e-5)
    assert np.all(np.diff(values) <= 0)


def test_training_freezes_only_after_stability_and_resumes_after_channel_change(tmp_path, monkeypatch):
    import elfcomm.training as training
    import elfcomm.plots as plots

    agents = []

    class FixedAgent:
        def __init__(self, *args, **kwargs):
            self.optimizer = SimpleNamespace(param_groups=[{}])
            self.updates = 0
            agents.append(self)

        def act(self, state, mask, epsilon=0.0):
            legal = np.flatnonzero(mask)
            return int(legal[0]) if legal.size else -1

        def remember(self, *args):
            pass

        def learn(self):
            self.updates += 1
            return 0.0

        def save(self, path):
            Path(path).write_bytes(b"test checkpoint")

    cfg = load_config("configs/shift_experiment.json")
    cfg["output_dir"] = str(tmp_path / "run")
    cfg["actions"]["frequency_hz"] = [85]
    cfg["actions"]["tx_power_kw"] = [62.5]
    cfg["training"].update(episodes=12, steps_per_episode=2, evaluation_interval=1)
    cfg["training"]["convergence"].update(enabled=True, window=2, min_phase_episodes=2)
    cfg["channel_shift"]["episode"] = 6
    monkeypatch.setattr(training, "DQNAgent", FixedAgent)
    monkeypatch.setattr(training, "tqdm", None)
    monkeypatch.setattr(plots, "plot_training", lambda *args: None)
    output = training.train(cfg, FakeTable(ber=1e-6))
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    rows = plots.read_csv(output / "training_history.csv")
    assert summary["episodes"] == 8 and summary["early_stopped"]
    assert [e["episode"] for e in summary["convergence_events"]] == [2, 8]
    assert agents[0].updates == 8
    assert rows[2]["learning_active"] == "False"
    assert rows[6]["learning_active"] == "True"
    assert float(rows[6]["learning_rate"]) == pytest.approx(1e-4)
    evaluations = plots.read_csv(output / "validation_history.csv")
    assert any(r["evaluation_event"] == "before_update" and r["diagnostic_only"] == "True" for r in evaluations)

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from elfcomm.channel import RelativeChannel
from elfcomm.config import Action, load_config
from elfcomm.environment import CommunicationEnvironment
from elfcomm.performance import evaluate_action, monotone_nonincreasing, utility


class FakeTable:
    def __init__(self, ber=1e-5, fer=0.1):
        self.ber = ber
        self.fer = fer
        self.calls = 0

    def lookup(self, modulation, rate, esn0_db):
        assert modulation in ("BPSK", "QPSK", "MSK")
        assert np.isfinite(esn0_db)
        self.calls += 1
        return {"ber_estimate": self.ber, "fer_estimate": self.fer,
                "table_status": "deterministic_test_fake", "reliability_certified": False}


@pytest.fixture
def cfg():
    return load_config()


def test_batch_channel_matches_single_action_metrics_and_masks(cfg):
    channel = RelativeChannel(cfg)
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["environment_noise_asd_pt_rt_hz"] = 0.25
    actions = [Action(f, m, 0.5, r) for f in (5.0, 7.0, 29.0)
               for m in ("BPSK", "QPSK", "MSK") for r in (0.25, 2 / 3)]
    bands = [{"low_hz": 5.9, "high_hz": 6.1, "mode": "hard"},
             {"low_hz": 6.95, "high_hz": 7.1, "mode": "soft", "interference_asd_pt_rt_hz": 0.4}]
    batch = channel.evaluate_many(actions, env, bands)
    for i, action in enumerate(actions):
        single = channel.evaluate(action, env, bands)
        for key, values in batch.items():
            if key == "legal":
                assert bool(values[i]) == single[key]
            else:
                assert values[i] == pytest.approx(single[key], rel=1e-11, abs=1e-11)


def test_environment_state_shape_mask_type_and_static_no_drift(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=30)
    reference = copy.deepcopy(cfg["link"]["reference_environment"])
    state, mask = environment.reset(reference, [], dynamic=False)
    assert state.shape == (284,)
    assert environment.state_dim == 284
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert mask.shape == (468,)
    assert mask.dtype == np.bool_
    index = int(np.flatnonzero(mask)[0])
    state_after, reward, done, new_mask, info = environment.step(index)
    assert environment.env == reference
    assert np.isfinite(reward)
    assert info["transmitted"] and not info["outage"]
    assert info["environment"] == reference
    assert state_after[-1] == 1.0
    assert not done
    np.testing.assert_array_equal(mask, new_mask)


def test_environment_rng_is_reproducible_and_independent_of_global_rng(cfg):
    first = CommunicationEnvironment(cfg, FakeTable(), seed=304)
    second = CommunicationEnvironment(cfg, FakeTable(), seed=304)
    for _ in range(12):
        np.testing.assert_array_equal(first.state(), second.state())
        assert first.env == second.env
        assert first.bands == second.bands
        index = int(np.flatnonzero(first.mask)[0]) if first.mask.any() else -1
        result1 = first.step(index)
        np.random.random(100)
        result2 = second.step(index)
        np.testing.assert_array_equal(result1[0], result2[0])
        assert result1[1] == result2[1]


def test_dynamic_environment_stays_within_ranges_including_noise(cfg):
    cfg["environment"]["step_drift_fraction"] = 0.5
    cfg["environment"]["noise_step_log_std"] = 2.0
    cfg["training"]["steps_per_episode"] = 70
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=111)
    start = copy.deepcopy(environment.env)
    for _ in range(70):
        index = int(np.flatnonzero(environment.mask)[0]) if environment.mask.any() else -1
        state, _, _, _, _ = environment.step(index)
        assert np.isfinite(state).all()
        for key, (low, high) in cfg["environment"]["ranges"].items():
            assert low <= environment.env[key] <= high
    assert start != environment.env


def test_all_occupied_yields_hold_outage_without_fictitious_ber(cfg):
    table = FakeTable()
    environment = CommunicationEnvironment(cfg, table, seed=2)
    full_band = [{"low_hz": 3.0, "high_hz": 30.0, "mode": "hard"}]
    _, mask = environment.reset(cfg["link"]["reference_environment"], full_band, dynamic=False)
    assert not mask.any()
    with pytest.raises(ValueError):
        environment.step(0)
    _, reward, _, mask, info = environment.step(-1)
    assert reward == cfg["objective"]["outage_reward"] == -1.0
    assert info["outage"] and not info["transmitted"]
    assert info["ber_estimate"] is None
    assert info["goodput_bps_estimate"] == 0.0
    assert table.calls == 0
    assert not mask.any()


def test_hold_is_rejected_when_transmission_available_and_done_requires_reset(cfg):
    cfg["training"]["steps_per_episode"] = 1
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=2)
    environment.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    with pytest.raises(ValueError):
        environment.step(-1)
    _, _, done, _, _ = environment.step(0)
    assert done
    with pytest.raises(RuntimeError):
        environment.step(0)


def test_occupancy_edge_mask_varies_with_symbol_rate(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=3)
    bands = [{"low_hz": 5.9, "high_hz": 6.1, "mode": "hard"}]
    environment.reset(cfg["link"]["reference_environment"], bands, dynamic=False)
    narrow = environment.actions.index(Action(5, "BPSK", 0.5, 0.25))
    wide = environment.actions.index(Action(5, "BPSK", 0.5, 2 / 3))
    assert environment.mask[narrow]
    assert not environment.mask[wide]
    with pytest.raises(ValueError):
        environment.step(wide)
    result = environment.action_metrics(narrow)
    assert result["legal"] is True


@pytest.mark.parametrize("modulation,bits_per_symbol", [("BPSK", 1), ("QPSK", 2), ("MSK", 1)])
def test_goodput_counts_crc_preamble_and_frame_errors(cfg, modulation, bits_per_symbol):
    action = Action(5, modulation, 0.75, 2 / 3)
    channel_metrics = RelativeChannel(cfg).evaluate(action, cfg["link"]["reference_environment"], [])
    metrics = evaluate_action(action, channel_metrics, FakeTable(fer=0.2), cfg)
    expected_k = 192
    expected_payload = 181
    expected_duration = (256 / bits_per_symbol + 16) / (2 / 3)
    assert metrics["information_bits_with_crc"] == expected_k
    assert metrics["payload_bits"] == expected_payload
    assert metrics["actual_code_rate"] == 0.75
    assert metrics["payload_code_rate"] == 181 / 256
    assert metrics["frame_duration_s"] == expected_duration
    assert metrics["goodput_bps_estimate"] == pytest.approx(expected_payload * 0.8 / expected_duration)
    assert metrics["nominal_information_rate_bps"] == pytest.approx((2 / 3) * bits_per_symbol * 181 / 256)
    assert not metrics["meets_rate_target_estimate"]
    assert not metrics["reliability_certified"]


def test_switching_penalty_is_applied_once(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=7)
    environment.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    first, second = 0, 1
    environment.step(first)
    _, switched_reward, _, _, switched = environment.step(second)
    expected_without_switch = utility(switched, cfg, switched=False)
    assert switched_reward == pytest.approx(expected_without_switch - cfg["objective"]["switch_penalty"])
    _, repeated_reward, _, _, _ = environment.step(second)
    assert repeated_reward == pytest.approx(expected_without_switch)


def test_weighted_monotone_fit_pools_only_inconsistent_blocks():
    result = monotone_nonincreasing([0.2, 0.05, 0.1, 0.001], [1, 1, 3, 1])
    np.testing.assert_allclose(result, [0.2, 0.0875, 0.0875, 0.001])
    assert np.all(np.diff(result) <= 0)


def test_outside_noise_is_part_of_state_even_when_center_frequency_not_overlapped(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=12)
    reference = cfg["link"]["reference_environment"]
    baseline, _ = environment.reset(reference, [], dynamic=False)
    bands = [{"low_hz": 5.3, "high_hz": 5.4, "mode": "soft", "interference_asd_pt_rt_hz": 1.0}]
    interfered, _ = environment.reset(reference, bands, dynamic=False)
    assert not np.array_equal(baseline, interfered)
    assert environment.mask.all()


def test_independent_cli_validation_uses_fixed_frame_budget_and_fer_bound(cfg, monkeypatch):
    import run

    seen = {}

    def simulate_link(modulation, rate, esn0_db, **settings):
        seen.update(settings)
        return {"fer": 0.02, "fer_upper95": 0.03}

    monkeypatch.setitem(sys.modules, "elfcomm.ber", SimpleNamespace(simulate_link=simulate_link))
    result = {"transmitted": True, "modulation": "QPSK", "code_rate": 0.75,
              "esn0_db": 10.0, "payload_bits": 181, "frame_duration_s": 216.0}
    report = run.validate_decision(cfg, result)
    assert seen["min_frames"] == seen["max_frames"] == cfg["validation"]["max_frames"]
    assert seen["seed"] == cfg["seed"] + 900001
    assert cfg["validation"]["min_frames"] == 100
    assert report["conservative_ber_upper95_via_fer"] == 0.03
    assert not report["reliability_certified"]
    assert report["measured_goodput_bps"] == pytest.approx(181 * 0.98 / 216)


def test_cli_decision_and_validation_all_blocked_skip_transmission(cfg, monkeypatch):
    import run

    class HoldAgent:
        def act(self, state, mask, epsilon):
            assert not mask.any()
            return -1

    def forbidden_simulation(*args, **kwargs):
        raise AssertionError("An outage is not an error-free transmission trial")

    monkeypatch.setitem(sys.modules, "elfcomm.ber", SimpleNamespace(simulate_link=forbidden_simulation))
    bands = [{"low_hz": 3, "high_hz": 30, "mode": "hard"}]
    result = run.decision(cfg, FakeTable(), HoldAgent(), cfg["link"]["reference_environment"], bands)
    assert result["decision_status"] == "no_legal_action_no_transmission"
    assert result["ber_estimate"] is None
    assert result["legal_action_count"] == 0
    report = run.validate_decision(cfg, result)
    assert report["status"] == "no_transmission"
    assert not report["reliability_certified"]


def test_documented_maximum_payload_throughput(cfg):
    action = Action(5, "QPSK", 0.75, 2 / 3)
    channel_metrics = RelativeChannel(cfg).evaluate(action, cfg["link"]["reference_environment"], [])
    metrics = evaluate_action(action, channel_metrics, FakeTable(fer=0), cfg)
    assert metrics["goodput_bps_estimate"] == pytest.approx(181 / 216)
    assert metrics["goodput_bps_estimate"] == pytest.approx(0.838, abs=0.0001)

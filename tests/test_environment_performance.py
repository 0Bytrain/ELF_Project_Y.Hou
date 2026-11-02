import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from elfcomm.channel import RelativeChannel
from elfcomm.config import Action, load_config, payload_rate_for
from elfcomm.environment import CommunicationEnvironment
from elfcomm.performance import evaluate_action, monotone_nonincreasing, utility
from elfcomm.training import checkpoint_contract, summarize
from elfcomm.config import fingerprint


class FakeTable:
    def __init__(self, ber=1e-5, fer=0.1):
        self.ber = ber
        self.fer = fer
        self.calls = 0

    def lookup_ber_many(self, coding_scheme, modulation, rate, esn0_db):
        self.calls += len(esn0_db)
        return np.full(len(esn0_db), self.ber)

    def lookup(self, coding_scheme, modulation, rate, esn0_db):
        assert coding_scheme in ("Polar", "CC", "BCH")
        assert modulation in ("BPSK", "QPSK", "MSK")
        assert np.isfinite(esn0_db)
        self.calls += 1
        return {"ber_estimate": self.ber, "fer_estimate": self.fer,
                "table_status": "deterministic_test_fake", "reliability_certified": False}


@pytest.fixture
def cfg():
    return load_config()


def make_action(cfg, frequency, modulation, code_rate, symbol_rate, power=500.0,
                scheme="Polar"):
    return Action(frequency, power, payload_rate_for(cfg, scheme, code_rate),
                  modulation, symbol_rate, code_rate, scheme)


def test_batch_channel_matches_single_action_metrics_and_masks(cfg):
    channel = RelativeChannel(cfg)
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["environment_noise_asd_pt_rt_hz"] = 0.25
    actions = [make_action(cfg, f, m, 0.5, r) for f in (17.0, 25.0, 33.0)
               for m in ("BPSK", "QPSK", "MSK") for r in (0.25, 2 / 3)]
    bands = [{"low_hz": 25.7, "high_hz": 25.75, "mode": "hard"},
             {"low_hz": 25.2, "high_hz": 25.3, "mode": "soft", "interference_asd_pt_rt_hz": 0.4}]
    batch = channel.evaluate_many(actions, env, bands)
    for i, action in enumerate(actions):
        single = channel.evaluate(action, env, bands)
        for key, values in batch.items():
            if key == "legal":
                assert bool(values[i]) == single[key]
            else:
                assert values[i] == pytest.approx(single[key], rel=1e-11, abs=1e-11)


def test_validation_uses_fixed_scenario_by_default(monkeypatch):
    import elfcomm.training as training

    seen = []

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, environment=None, occupied_bands=None, dynamic=True):
            seen.append(dynamic)
            return np.zeros(2, dtype=np.float32), np.ones(1, dtype=bool)

        def step(self, action):
            return np.zeros(2, dtype=np.float32), 0.0, True, np.ones(1, dtype=bool), {
                "transmitted": False, "outage": True, "ber_estimate": None,
                "goodput_bps_estimate": 0.0, "reliability_certified": False,
                "reward": 0.0,
            }

    class FakeAgent:
        def act(self, state, mask, epsilon=0.0):
            return 0

    monkeypatch.setattr(training, "CommunicationEnvironment", FakeEnv)
    training.evaluate_policy(FakeAgent(), {"seed": 1, "training": {"evaluation_episodes": 1}}, None)
    assert seen == [False]


def test_validation_uses_reference_channel_when_no_fixed_environment(cfg):
    import elfcomm.training as training

    class FirstLegalAgent:
        def act(self, state, mask, epsilon=0.0):
            return int(np.flatnonzero(mask)[0])

    _, records = training.evaluate_policy(FirstLegalAgent(), cfg, FakeTable(), episode_count=2)
    reference = cfg["link"]["reference_environment"]
    assert records
    assert all(record["environment"] == reference for record in records)


def test_environment_state_shape_mask_type_and_static_no_drift(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=30)
    reference = copy.deepcopy(cfg["link"]["reference_environment"])
    state, mask = environment.reset(reference, [], dynamic=False)
    assert state.shape == (1145,)
    assert environment.state_dim == 1145
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert mask.shape == (74844,)
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


def test_random_soft_occupancy_is_reachable_when_probability_is_lowered(cfg):
    cfg["environment"]["random_hard_probability"] = 0.0
    cfg["environment"]["random_band_count_max"] = 1
    cfg["environment"]["random_occupancy_min_hz"] = 60.0
    saw_low_soft = False
    for seed in range(80):
        environment = CommunicationEnvironment(cfg, FakeTable(), seed=seed)
        for band in environment.random_bands:
            if band["mode"] == "soft" and band["high_hz"] < 60.0:
                saw_low_soft = True
    assert saw_low_soft


def test_random_occupancy_is_hard_and_restricted_to_high_band(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=42)
    assert environment.random_bands
    for band in environment.random_bands:
        assert band["mode"] == "hard"
        assert band["high_hz"] - band["low_hz"] == pytest.approx(1.2)
        assert band["low_hz"] >= cfg["environment"]["random_occupancy_min_hz"] - 0.6 - 1e-9
    for index, action in enumerate(environment.actions):
        if action.frequency_hz < cfg["environment"]["random_occupancy_min_hz"]:
            assert environment.mask[index]


def test_all_occupied_yields_hold_outage_without_fictitious_ber(cfg):
    table = FakeTable()
    environment = CommunicationEnvironment(cfg, table, seed=2)
    initial_calls = table.calls
    full_band = [{"low_hz": 3.0, "high_hz": 300.5, "mode": "hard"}]
    _, mask = environment.reset(cfg["link"]["reference_environment"], full_band, dynamic=False)
    assert not mask.any()
    with pytest.raises(ValueError):
        environment.step(0)
    _, reward, _, mask, info = environment.step(-1)
    assert reward == cfg["objective"]["outage_reward"] == -1.0
    assert info["outage"] and not info["transmitted"]
    assert info["ber_estimate"] is None
    assert info["goodput_bps_estimate"] == 0.0
    assert table.calls == initial_calls
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
    bands = [{"low_hz": 25.7, "high_hz": 25.75, "mode": "hard"}]
    environment.reset(cfg["link"]["reference_environment"], bands, dynamic=False)
    narrow = environment.actions.index(make_action(cfg, 25, "BPSK", 0.5, 0.25))
    wide = environment.actions.index(make_action(cfg, 25, "BPSK", 0.5, 2 / 3))
    assert environment.mask[narrow]
    assert not environment.mask[wide]
    with pytest.raises(ValueError):
        environment.step(wide)
    result = environment.action_metrics(narrow)
    assert result["legal"] is True


@pytest.mark.parametrize("modulation,bits_per_symbol", [("BPSK", 1), ("QPSK", 2), ("MSK", 1)])
def test_goodput_counts_crc_preamble_and_frame_errors(cfg, modulation, bits_per_symbol):
    action = make_action(cfg, 25, modulation, 0.75, 2 / 3)
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
    expected_goodput = expected_payload * 0.8 / expected_duration
    assert metrics["meets_rate_target_estimate"] == (expected_goodput >= cfg["objective"]["target_goodput_bps"])
    assert not metrics["reliability_certified"]


def test_switching_penalty_is_applied_once(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=7)
    environment.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    legal = np.flatnonzero(environment.mask)
    first, second = int(legal[0]), int(legal[1])
    environment.step(first)
    _, switched_reward, _, _, switched = environment.step(second)
    expected_without_switch = utility(switched, cfg, switched=False)
    assert switched_reward == pytest.approx(expected_without_switch - cfg["objective"]["switch_penalty"])
    _, repeated_reward, _, _, _ = environment.step(second)
    assert repeated_reward == pytest.approx(expected_without_switch)


def test_utility_uses_fixed_ber_target_and_prefers_lower_feasible_power(cfg):
    base = {"legal": True, "goodput_bps_estimate": cfg["objective"]["target_goodput_bps"],
            "ber_estimate": 1e-5, "tx_power_kw": 500.0}
    high = utility(base, cfg)
    low = utility({**base, "tx_power_kw": 62.5}, cfg)
    assert low > high
    hot = {**base, "ber_estimate": 0.2, "tx_power_kw": 62.5}
    assert utility(hot, cfg, baseline_ber=1e-5) == utility(hot, cfg)
    assert utility(hot, cfg) < high


def test_weighted_monotone_fit_pools_only_inconsistent_blocks():
    result = monotone_nonincreasing([0.2, 0.05, 0.1, 0.001], [1, 1, 3, 1])
    np.testing.assert_allclose(result, [0.2, 0.0875, 0.0875, 0.001])
    assert np.all(np.diff(result) <= 0)


def test_outside_noise_is_part_of_state_even_when_center_frequency_not_overlapped(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=12)
    reference = cfg["link"]["reference_environment"]
    baseline, _ = environment.reset(reference, [], dynamic=False)
    bands = [{"low_hz": 25.2, "high_hz": 25.3, "mode": "soft", "interference_asd_pt_rt_hz": 1.0}]
    interfered, _ = environment.reset(reference, bands, dynamic=False)
    assert not np.array_equal(baseline, interfered)
    assert environment.mask.any()


def test_independent_cli_validation_uses_fixed_frame_budget_and_fer_bound(cfg, monkeypatch):
    import run

    cfg["objective"]["target_ber"] = 1e-4

    seen = {}

    def simulate_link(coding_scheme, modulation, rate, esn0_db, **settings):
        seen.update(settings, coding_scheme=coding_scheme, modulation=modulation,
                    rate=rate, esn0_db=esn0_db)
        return {"fer": 0.02, "fer_upper95": 0.03}

    monkeypatch.setitem(sys.modules, "elfcomm.ber", SimpleNamespace(simulate_link=simulate_link))
    result = {"transmitted": True, "coding_scheme": "Polar", "modulation": "QPSK",
              "code_rate": 0.75, "esn0_db": 10.0, "payload_bits": 181,
              "frame_duration_s": 216.0}
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
    bands = [{"low_hz": 3, "high_hz": 300.5, "mode": "hard"}]
    result = run.decision(cfg, FakeTable(), HoldAgent(), cfg["link"]["reference_environment"], bands)
    assert result["decision_status"] == "no_legal_action_no_transmission"
    assert result["ber_estimate"] is None
    assert result["legal_action_count"] == 0
    assert result["distance_gain_frequency_hz"] == cfg["link"]["reference_frequency_hz"]
    report = run.validate_decision(cfg, result)
    assert report["status"] == "no_transmission"
    assert not report["reliability_certified"]


def test_cli_decision_reports_distance_gain_at_selected_frequency(cfg):
    import run

    class FirstLegalAgent:
        def act(self, state, mask, epsilon):
            return int(np.flatnonzero(mask)[0])

    scenario = copy.deepcopy(cfg["link"]["reference_environment"])
    scenario["horizontal_distance_m"] = 1_200_000.0
    result = run.decision(cfg, FakeTable(), FirstLegalAgent(), scenario, [])
    selected = result["frequency_hz"]
    assert result["distance_gain_frequency_hz"] == selected
    expected = RelativeChannel(cfg).distance_gain_db(selected, scenario["horizontal_distance_m"])
    assert result["distance_gain_db"] == pytest.approx(expected)


def test_summarize_reports_conditioning_snr_and_uses_transmitted_denominator():
    records = [
        {"transmitted": True, "payload_bits": 10, "frame_duration_s": 1.0, "reward": 0.0,
         "ber_estimate": 1e-5, "goodput_bps_estimate": 0.8, "meets_ber_target_estimate": True,
         "esn0_db": 5.0, "snr_db": 3.0, "power_changed": False, "tx_power_kw": 500.0},
        {"transmitted": True, "payload_bits": 10, "frame_duration_s": 1.0, "reward": 0.0,
         "ber_estimate": 1e-5, "goodput_bps_estimate": 0.8, "meets_ber_target_estimate": True,
         "esn0_db": 7.0, "snr_db": 5.0, "power_changed": True, "tx_power_kw": 250.0},
        {"transmitted": False, "outage": True, "ber_estimate": None,
         "goodput_bps_estimate": 0.0, "reward": -1.0},
    ]
    summary = summarize(records)
    assert summary["ber_target_met_fraction_estimate"] == 1.0
    assert summary["power_switch_fraction"] == 0.5
    assert summary["mean_esn0_db"] == pytest.approx(6.0)
    assert summary["mean_snr_db"] == pytest.approx(4.0)


def test_environment_rejects_scenario_outside_declared_ranges(cfg):
    environment = CommunicationEnvironment(cfg, FakeTable(), seed=88)
    scenario = copy.deepcopy(cfg["link"]["reference_environment"])
    scenario["environment_noise_asd_pt_rt_hz"] = 10.0
    with pytest.raises(ValueError, match="outside configured range"):
        environment.reset(scenario, [], dynamic=False)


def test_cli_scenario_keeps_random_occupancy_enabled_when_no_bands_given(cfg):
    import run

    args = SimpleNamespace(scenario=None, noise_asd=None, depth=None, distance=None,
                           ice_thickness=None, ice_temperature=None, water_conductivity=None)
    _, bands = run.scenario_from_args(cfg, args)
    assert bands is None


def test_checkpoint_contract_tracks_channel_and_mask_changes(cfg):
    baseline = fingerprint(checkpoint_contract(cfg))
    changed = copy.deepcopy(cfg)
    changed["link"]["receiver_noise_rms_pt"] *= 1.01
    assert fingerprint(checkpoint_contract(changed)) != baseline


def test_documented_maximum_payload_throughput(cfg):
    action = make_action(cfg, 25, "QPSK", 0.75, 2 / 3)
    channel_metrics = RelativeChannel(cfg).evaluate(action, cfg["link"]["reference_environment"], [])
    metrics = evaluate_action(action, channel_metrics, FakeTable(fer=0), cfg)
    assert metrics["goodput_bps_estimate"] == pytest.approx(181 / 216)
    assert metrics["goodput_bps_estimate"] == pytest.approx(0.838, abs=0.0001)
    assert metrics["waveform_mode"][0] == "QPSK"
    assert metrics["waveform_mode"][1] == pytest.approx(2 / 3)
    assert metrics["waveform_mode_index"] == 5

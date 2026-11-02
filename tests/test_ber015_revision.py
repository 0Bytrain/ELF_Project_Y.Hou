import copy

import numpy as np
import pytest

from elfcomm.config import fingerprint, load_config, save_json
from elfcomm.convergence import phase_assessment, tail_statistics
from elfcomm.environment import CommunicationEnvironment
from elfcomm.performance import PerformanceTable, row_key, table_specification, utility


def test_lower_power_feasible_action_beats_all_higher_power_tie_breaks():
    cfg = load_config()
    limit = cfg["objective"]["target_ber"]
    base = {"legal": True, "ber_estimate": limit * 0.99, "tx_power_kw": 62.5,
            "goodput_bps_estimate": 0.0}
    lower = utility(base, cfg, switched=True)
    higher = utility({**base, "tx_power_kw": 125, "goodput_bps_estimate": 100.0}, cfg)
    assert lower > higher
    assert utility({**base, "ber_estimate": limit * 1.01}, cfg) < utility(
        {**base, "tx_power_kw": 500}, cfg)
    assert utility(base, cfg, baseline_ber=1e-8) == utility(base, cfg)


def make_table(tmp_path, counts):
    cfg = copy.deepcopy(load_config())
    cfg["actions"]["coding_mode"] = [["Polar", 0.5]]
    cfg["actions"]["modulation"] = ["BPSK"]
    cfg["ber_table"]["esn0_db"] = [0, 2]
    spec = table_specification(cfg)
    rows = {row_key("Polar", "BPSK", 0.5, snr): {
        "bit_errors": errors, "total_bits": bits,
        "frame_errors": min(errors, 10), "total_frames": 10,
    } for snr, (errors, bits) in zip([0, 2], counts)}
    path = tmp_path / "counts.json"
    save_json(path, {"signature": fingerprint(spec), "specification": spec,
                     "complete": True, "rows": rows})
    return PerformanceTable(cfg, path)


def test_zero_error_knots_do_not_acquire_a_payload_dependent_fake_ber(tmp_path):
    table = make_table(tmp_path, [(0, 100), (0, 200)])
    for snr in (0, 1, 2, 3):
        result = table.lookup("Polar", "BPSK", 0.5, snr)
        assert result["ber_estimate"] == 0
        assert result["fer_estimate"] == 0
        assert result["ber_resolution_limited"]
        assert not result["reliability_certified"]
    assert table.lookup("Polar", "BPSK", 0.5, 3)["ber_resolution_limit"] == 1 / 200


def test_interpolation_can_reach_zero_without_log_of_zero(tmp_path):
    table = make_table(tmp_path, [(20, 100), (0, 200)])
    result = table.lookup("Polar", "BPSK", 0.5, 1)
    assert result["ber_estimate"] == pytest.approx(0.1)
    assert not result["ber_resolution_limited"]
    assert result["ber_interpolation_uses_zero_counts"]


def records(values):
    return [{"episode": i + 1, "ber_estimate": v, "outage_fraction": 0.0}
            for i, v in enumerate(values)]


def test_only_final_window_can_establish_convergence():
    stats = tail_statistics(records([0.05] * 10 + [0.04, 0.14] * 5), window=10)
    assert not stats["converged"]
    assert stats["start"] == 11
    assert stats["end"] == 20


def test_absolute_target_replaces_relative_pre_post_comparison():
    assessment = phase_assessment(records([0.05] * 10 + [0.14] * 10), 10, 0.15)
    assert assessment["criterion_met"]
    assessment = phase_assessment(records([0.16] * 10 + [0.14] * 10), 10, 0.15)
    assert not assessment["criterion_met"]
    assert not assessment["both_phases_target_met_estimate"]


def test_zero_count_plateau_does_not_prove_resolved_ber_convergence():
    stats = tail_statistics(records([0.0] * 10), target_ber=0.15)
    assert not stats["converged"]
    assert stats["target_met"]
    assert stats["reason"] == "below_table_resolution"


def test_outages_or_missing_values_cannot_be_silently_skipped():
    rows = records([0.05] * 20)
    rows[-1]["ber_estimate"] = None
    assert not tail_statistics(rows)["converged"]
    rows[-1]["ber_estimate"] = 0.05
    rows[-1]["outage_fraction"] = 0.2
    stats = tail_statistics(rows, target_ber=0.15)
    assert not stats["converged"]
    assert not stats["target_met"]


def test_power_cap_keeps_legal_frequency_features_when_lower_powers_are_available():
    cfg = load_config()
    cfg["decision"]["ber_feasibility_filter"] = False
    env = CommunicationEnvironment(cfg, object())
    state, _ = env.reset(cfg["link"]["reference_environment"], [], dynamic=False)
    env.set_power_cap(250)
    nf = len(cfg["actions"]["frequency_hz"])
    nw = len(cfg["actions"]["waveform_mode"])
    legal = env.state()[6 + 3 * nf:6 + 3 * nf + nf * nw]
    assert env.mask.any()
    np.testing.assert_array_equal(legal, state[6 + 3 * nf:6 + 3 * nf + nf * nw])


def test_guard_distance_only_matters_until_emitted_band_is_clear():
    cfg = load_config()
    cfg["decision"]["ber_feasibility_filter"] = False
    env = CommunicationEnvironment(cfg, object())
    reference = cfg["link"]["reference_environment"]
    band = [{"low_hz": 24.4, "high_hz": 25.6, "mode": "hard"}]
    env.reset(reference, band, dynamic=False)
    assert all(not env.mask[i] for i, a in enumerate(env.actions) if a.frequency_hz == 25)
    assert any(env.mask[i] for i, a in enumerate(env.actions) if a.frequency_hz == 33)
    assert any(env.mask[i] for i, a in enumerate(env.actions) if a.frequency_hz == 41)

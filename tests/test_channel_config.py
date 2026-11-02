import copy

import numpy as np
import pytest

from elfcomm.antenna import AntennaModel
from elfcomm.channel import (
    RelativeChannel, action_allowed, magnetic_transfer, material, occupied_bandwidth,
)
from elfcomm.config import (Action, action_shape, build_actions, load_config,
                            payload_rate_for, quick_config, validate_config)
from elfcomm.propagation import PropagationModel
from elfcomm.range_model import RangeModel


@pytest.fixture
def cfg():
    return load_config()


def make_action(cfg, frequency, modulation, code_rate, symbol_rate, power=500.0,
                scheme="Polar"):
    return Action(frequency, power, payload_rate_for(cfg, scheme, code_rate),
                  modulation, symbol_rate, code_rate, scheme)


def test_reference_calibration_matches_lwpc_anchor_chain(cfg):
    channel = RelativeChannel(cfg)
    link = cfg["link"]
    env = copy.deepcopy(link["reference_environment"])
    signal = channel.signal_mean_square(link["reference_frequency_hz"], env)
    noise = channel.background_noise_psd(link["reference_frequency_hz"], env)
    ratio = signal / (noise * link["reference_noise_bandwidth_hz"])
    assert np.sqrt(signal) == pytest.approx(1.5906, abs=0.001)
    assert np.sqrt(noise * link["reference_noise_bandwidth_hz"]) == pytest.approx(np.sqrt(0.11), abs=1e-4)
    assert 10 * np.log10(ratio) == pytest.approx(13.62, abs=0.02)
    assert channel.reference_snr_db == pytest.approx(13.62, abs=0.02)
    assert abs(channel.reference_transfer) == pytest.approx(0.67358, abs=1e-4)
    # 100 km cross-check with the image scalar anchor (81.07 dB -> 25.41 pT full-wave).
    # The Lu attenuation formula is restricted to d >= propagation_min_km, so this
    # near-field check deliberately uses the LWPC table as the distance factor.
    table_gain = RangeModel(link).gain_db(100_000.0)
    cross_check = channel.source_field_rms_pt * abs(channel.reference_transfer) * 10 ** (table_gain / 20)
    assert cross_check == pytest.approx(25.41, rel=1e-3)


def test_noise_rms_scales_with_sqrt_bandwidth(cfg):
    channel = RelativeChannel(cfg)
    env = cfg["link"]["reference_environment"]
    narrow = channel.evaluate(make_action(cfg, 17, "BPSK", 0.5, 0.25), env, [])
    wide = channel.evaluate(make_action(cfg, 17, "BPSK", 0.5, 0.5), env, [])
    assert wide["noise_rms_pt"] / narrow["noise_rms_pt"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert wide["rx_field_rms_pt_assumed"] == pytest.approx(narrow["rx_field_rms_pt_assumed"])
    assert wide["snr_db"] == pytest.approx(narrow["snr_db"] - 10 * np.log10(2), abs=0.01)


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "MSK"])
def test_esn0_uses_symbol_rate_and_noise_psd(cfg, modulation):
    channel = RelativeChannel(cfg)
    action = make_action(cfg, 25, modulation, 0.5, 0.5)
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["environment_noise_asd_pt_rt_hz"] = 0.3
    result = channel.evaluate(action, env, [])
    expected = result["rx_field_rms_pt_assumed"] ** 2 / (
        result["effective_noise_psd_pt2_hz"] * action.symbol_rate
    )
    assert 10 ** (result["esn0_db"] / 10) == pytest.approx(expected)
    assert result["esn0_db"] - result["snr_db"] == pytest.approx(
        10 * np.log10(result["occupied_bandwidth_hz"] / action.symbol_rate)
    )


@pytest.mark.parametrize("temperature", [-25.0, -10.0, -2.0])
def test_temperature_and_frequency_extremes_are_finite(cfg, temperature):
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["ice_temperature_c"] = temperature
    field = magnetic_transfer(np.array([3, 5, 15, 29, 30]), env, cfg["link"])
    assert np.isfinite(field).all()
    assert np.all(np.abs(field) > 0)


def test_zero_thickness_ice_equals_air_water_magnetic_fresnel(cfg):
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["ice_thickness_m"] = 0.0
    env["receiver_depth_m"] = 0.0
    frequency = np.array([5.0, 13.0, 29.0])
    _, z_air = material(frequency, 1.0)
    _, z_water = material(frequency, cfg["link"]["seawater_relative_permittivity"], env["seawater_conductivity_s_m"])
    expected = 2 * z_air / (z_air + z_water)
    actual = magnetic_transfer(frequency, env, cfg["link"])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_deeper_receiver_has_exact_conductive_medium_attenuation(cfg):
    shallow = copy.deepcopy(cfg["link"]["reference_environment"])
    deep = copy.deepcopy(shallow)
    deep["receiver_depth_m"] += 20.0
    frequency = np.array([5.0, 15.0, 29.0])
    ratio = np.abs(magnetic_transfer(frequency, deep, cfg["link"]) / magnetic_transfer(frequency, shallow, cfg["link"]))
    gamma, _ = material(frequency, cfg["link"]["seawater_relative_permittivity"], deep["seawater_conductivity_s_m"])
    np.testing.assert_allclose(ratio, np.exp(-gamma.real * 20.0))
    assert np.all(ratio < 1.0)


def test_lu_attenuation_matches_textbook_closed_forms(cfg):
    day = PropagationModel(cfg["link"])
    assert day.beta_db_per_mm(30.0) == pytest.approx(0.64681, rel=1e-4)
    assert day.beta_db_per_mm(300.0) == pytest.approx(5.39057, rel=1e-4)
    assert day.gain_db(30.0, 1_950_000.0) == pytest.approx(0.0, abs=1e-9)
    assert day.gain_db(30.0, 1_200_000.0) == pytest.approx(4.70218, abs=0.01)
    assert day.gain_db(30.0, 2_950_000.0) == pytest.approx(-4.24256, abs=0.01)

    night_link = copy.deepcopy(cfg["link"])
    night_link["ionosphere_mode"] = "night"
    night_link["ionosphere"].update(H_km=90.0, sigma0_km=2.5, hE_km=90.0, sigmaE_s_m=8e-6)
    night = PropagationModel(night_link)
    assert night.beta_db_per_mm(30.0) == pytest.approx(0.55569, rel=1e-4)
    assert night.beta_db_per_mm(300.0) == pytest.approx(2.66699, rel=1e-4)
    # 30 Hz formula reproduces the LWPC shape within ~1.5 dB over the training range.
    table = RangeModel(cfg["link"])
    for distance in (1_200_000.0, 2_950_000.0):
        assert day.gain_db(30.0, distance) == pytest.approx(table.gain_db(distance), abs=1.5)
    with pytest.raises(ValueError):
        day.gain_db(30.0, 100_000.0)


def test_antenna_model_uses_liu_current_moment_and_lu_tables(cfg):
    antenna = AntennaModel(cfg["link"])
    assert antenna.feed_current_a(30.0, 500.0) == pytest.approx(170.5565, rel=1e-4)
    assert antenna.current_moment_a_m(30.0, 500.0) == pytest.approx(30e6, rel=1e-9)
    assert antenna.current_moment_a_m(30.0, 250.0) == pytest.approx(30e6 / np.sqrt(2), rel=1e-9)
    assert antenna.effective_length_m == pytest.approx(175.88e3, rel=1e-3)
    assert antenna.input_resistance_ohm(100.0) == pytest.approx(26.2)
    assert antenna.radiation_resistance_ohm(100.0) == pytest.approx(4.4e-4)
    assert antenna.radiation_efficiency(100.0) == pytest.approx(4.4e-4 / 26.2)
    assert antenna.radiated_power_w(100.0, 500.0) == pytest.approx(8.4, rel=1e-3)
    si = antenna.field_uv_m(30.0, 1_950_000.0, 500.0, "si")
    engineering = antenna.field_uv_m(30.0, 1_950_000.0, 500.0, "engineering")
    assert 20 * np.log10(si) == pytest.approx(36.084, abs=0.01)
    assert 20 * np.log10(engineering) == pytest.approx(36.084, abs=0.01)
    # Both branches must return the same physical uV/m field.  The old
    # engineering branch returned a value 30 dB too high.
    assert engineering / si == pytest.approx(1.0, rel=1e-12)


def test_optional_current_moment_source_model(cfg):
    changed = copy.deepcopy(cfg)
    changed["link"]["source_model"] = "current_moment"
    channel = RelativeChannel(changed)
    antenna = AntennaModel(changed["link"])
    env = copy.deepcopy(changed["link"]["reference_environment"])
    local = abs(magnetic_transfer(changed["link"]["reference_frequency_hz"], env, changed["link"]))
    expected = antenna.field_rms_pt(changed["link"]["reference_frequency_hz"],
                                    env["horizontal_distance_m"],
                                    changed["link"]["reference_tx_power_kw"]) * local
    assert channel.reference_signal_rms_pt == pytest.approx(expected)
    assert channel.source_model == "current_moment"
    full = channel.evaluate(make_action(changed, 25, "BPSK", 0.5, 0.25), env, [])
    half = channel.evaluate(make_action(changed, 25, "BPSK", 0.5, 0.25, power=250.0), env, [])
    assert half["esn0_db"] - full["esn0_db"] == pytest.approx(-3.0103, abs=1e-3)


def test_configuration_rejects_invalid_antenna(cfg):
    cfg["link"]["antenna"]["current_moment_a_m"] = 0
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = load_config()
    cfg["link"]["source_model"] = "unknown"
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_range_model_uses_lwpc_table_and_reference_anchor(cfg):
    model = RangeModel(cfg["link"])
    assert model.gain_db(1_950_000.0) == pytest.approx(0.0, abs=1e-9)
    assert model.gain_db(100_000.0) == pytest.approx(24.07, abs=0.02)
    assert model.gain_db(1_200_000.0) == pytest.approx(5.31, abs=0.02)
    assert model.gain_db(2_600_000.0) == pytest.approx(model.gain_db(2_550_000.0), abs=1e-9)
    assert model.power_ratio(100_000.0) == pytest.approx(10 ** 2.407, rel=1e-3)
    assert model.field_ratio(1_950_000.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        model.gain_db(4_000_000.0)


def test_hard_occupancy_mask_accounts_for_waveform_width_and_guard(cfg):
    band = [{"low_hz": 25.7, "high_hz": 25.75, "mode": "hard"}]
    slow = make_action(cfg, 25.0, "BPSK", 0.5, 0.25)
    fast = make_action(cfg, 25.0, "BPSK", 0.5, 2 / 3)
    assert action_allowed(slow, band, cfg["link"])
    assert not action_allowed(fast, band, cfg["link"])
    link_no_guard = copy.deepcopy(cfg["link"])
    link_no_guard["guard_band_hz"] = 0.0
    assert action_allowed(fast, band, link_no_guard)


def test_soft_band_without_asd_is_zero_in_single_and_batch(cfg):
    channel = RelativeChannel(cfg)
    action = make_action(cfg, 25, "QPSK", 0.5, 0.5)
    band = [{"low_hz": 25.2, "high_hz": 25.3, "mode": "soft"}]
    single = channel.evaluate(action, cfg["link"]["reference_environment"], band)
    batch = channel.evaluate_many([action], cfg["link"]["reference_environment"], band)
    assert batch["effective_noise_psd_pt2_hz"][0] == pytest.approx(single["effective_noise_psd_pt2_hz"])


def test_soft_interference_integrates_only_overlap_bandwidth(cfg):
    channel = RelativeChannel(cfg)
    action = make_action(cfg, 25, "QPSK", 0.5, 0.5)
    env = cfg["link"]["reference_environment"]
    baseline = channel.evaluate(action, env, [])
    band = [{"low_hz": 25.2, "high_hz": 25.3, "mode": "soft", "interference_asd_pt_rt_hz": 0.3}]
    interfered = channel.evaluate(action, env, band)
    expected_psd_increase = 0.1 * 0.3 ** 2 / occupied_bandwidth(action, cfg["link"])
    assert interfered["effective_noise_psd_pt2_hz"] - baseline["effective_noise_psd_pt2_hz"] == pytest.approx(expected_psd_increase)
    assert interfered["legal"]
    assert interfered["esn0_db"] < baseline["esn0_db"]


def test_whole_band_occupied_blocks_every_action(cfg):
    bands = [{"low_hz": 3.0, "high_hz": 300.5, "mode": "hard"}]
    assert not any(action_allowed(action, bands, cfg["link"]) for action in build_actions(cfg))


def test_power_scaling_is_3db_per_step(cfg):
    channel = RelativeChannel(cfg)
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    full = channel.evaluate(make_action(cfg, 25, "BPSK", 0.5, 0.25), env, [])
    half = channel.evaluate(make_action(cfg, 25, "BPSK", 0.5, 0.25, power=250.0), env, [])
    assert half["esn0_db"] - full["esn0_db"] == pytest.approx(-3.0103, abs=1e-3)
    assert half["rx_field_rms_pt_assumed"] / full["rx_field_rms_pt_assumed"] == pytest.approx(10 ** (-3.0103 / 20), rel=1e-4)
    # The same linear power law must hold away from the reference distance;
    # this catches a dB/km vs dB/Mm unit error in the attenuation term.
    env["horizontal_distance_m"] = 1_200_000.0
    far_full = channel.evaluate(make_action(cfg, 25, "BPSK", 0.5, 0.25), env, [])
    far_half = channel.evaluate(make_action(cfg, 25, "BPSK", 0.5, 0.25, power=250.0), env, [])
    assert far_half["esn0_db"] - far_full["esn0_db"] == pytest.approx(-3.0103, abs=1e-3)


def test_evaluate_and_batch_agree_at_formula_distance(cfg):
    channel = RelativeChannel(cfg)
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    env["horizontal_distance_m"] = 2_950_000.0
    action = make_action(cfg, 100, "QPSK", 0.5, 0.5, power=125.0)
    single = channel.evaluate(action, env, [])
    batch = channel.evaluate_many([action], env, [])
    for key in ("rx_field_rms_pt_assumed", "snr_db", "esn0_db"):
        assert batch[key][0] == pytest.approx(single[key], rel=1e-12)


def test_polarization_is_not_an_action_axis_and_coding_is_one_axis(cfg):
    assert "polarization" not in cfg["actions"]
    assert "polarization_model" not in cfg["link"]
    assert "payload_rate" not in cfg["actions"]
    assert "code_rate" not in cfg["actions"]
    assert cfg["actions"]["coding_mode"] == [
        ["Polar", 0.3333333333333333], ["Polar", 0.5],
        ["Polar", 0.6666666666666666], ["Polar", 0.75],
        ["CC", 0.5], ["CC", 0.6666666666666666], ["CC", 0.75],
        ["BCH", 0.3568627450980392], ["BCH", 0.5137254901960784],
        ["BCH", 0.6705882352941176], ["BCH", 0.7490196078431373],
    ]
    assert cfg["actions"]["waveform_mode"] == [
        ["BPSK", 0.25], ["BPSK", 0.5], ["BPSK", 0.6666666666666666],
        ["QPSK", 0.25], ["QPSK", 0.5], ["QPSK", 0.6666666666666666],
        ["MSK", 0.25], ["MSK", 0.5], ["MSK", 0.6666666666666666],
    ]


def test_action_product_and_quick_config_are_independent(cfg):
    assert action_shape(cfg) == (54, 14, 9, 11)
    assert len(build_actions(cfg)) == 74844
    fast = quick_config(cfg)
    assert fast["polar"]["n"] == 64
    assert action_shape(fast) == (10, 2, 9, 7)
    assert len(build_actions(fast)) == 1260
    assert cfg["polar"]["n"] == 256
    assert "SMOKE" in fast["run_label"]


@pytest.mark.parametrize("invalid", ["duplicate_axis", "not_power_two", "no_payload", "bad_frequency", "under_seabed"])
def test_configuration_rejects_invalid_inputs(cfg, invalid):
    if invalid == "duplicate_axis":
        cfg["actions"]["frequency_hz"].append(17)
    elif invalid == "not_power_two":
        cfg["polar"]["n"] = 192
    elif invalid == "no_payload":
        cfg["actions"]["coding_mode"] = [["Polar", 0.01]]
    elif invalid == "bad_frequency":
        cfg["actions"]["frequency_hz"] = [301]
    elif invalid == "under_seabed":
        cfg["environment"]["ranges"]["receiver_depth_m"] = [30, 150]
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("legacy", ["fixed_tx_power_kw", "range_spreading_exponent", "reference_snr_db"])
def test_configuration_rejects_legacy_link_fields(cfg, legacy):
    cfg["link"][legacy] = 1.0
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("invalid", ["nan_rate", "infinite_noise", "reference_under_seabed", "negative_reference_noise", "absolute_zero_ice", "overwide_to_dc", "nonfinite_occupancy", "lwpc_non30", "negative_preamble"])
def test_configuration_safety_regressions(cfg, invalid):
    if invalid == "nan_rate":
        cfg["actions"]["symbol_rate"] = [float("nan")]
    elif invalid == "infinite_noise":
        cfg["link"]["receiver_noise_rms_pt"] = float("inf")
    elif invalid == "reference_under_seabed":
        cfg["link"]["reference_environment"]["receiver_depth_m"] = 150
    elif invalid == "negative_reference_noise":
        cfg["link"]["reference_environment"]["environment_noise_asd_pt_rt_hz"] = -1
    elif invalid == "absolute_zero_ice":
        cfg["environment"]["ranges"]["ice_temperature_c"] = [-273.15, -2]
    elif invalid == "overwide_to_dc":
        cfg["actions"]["symbol_rate"] = [100.0]
    elif invalid == "nonfinite_occupancy":
        cfg["environment"]["occupied_bands"] = [{"low_hz": 5.0, "high_hz": float("nan"), "mode": "hard"}]
    elif invalid == "lwpc_non30":
        cfg["link"]["propagation_model"] = "lwpc_table"
        cfg["link"]["reference_frequency_hz"] = 100.0
    elif invalid == "negative_preamble":
        cfg["objective"]["preamble_symbols"] = -1
    with pytest.raises(ValueError):
        validate_config(cfg)

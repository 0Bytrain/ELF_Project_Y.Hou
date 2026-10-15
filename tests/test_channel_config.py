import copy

import numpy as np
import pytest

from elfcomm.channel import (
    RelativeChannel, action_allowed, magnetic_transfer, material, occupied_bandwidth,
)
from elfcomm.config import Action, action_shape, build_actions, load_config, quick_config, validate_config


@pytest.fixture
def cfg():
    return load_config()


def test_reference_calibration_matches_configured_anchor(cfg):
    channel = RelativeChannel(cfg)
    link = cfg["link"]
    env = copy.deepcopy(link["reference_environment"])
    signal = channel.signal_mean_square(link["reference_frequency_hz"], env)
    noise = channel.background_noise_psd(link["reference_frequency_hz"], env)
    ratio = signal / (noise * link["reference_noise_bandwidth_hz"])
    assert 10 * np.log10(ratio) == pytest.approx(link["reference_snr_db"])
    assert np.sqrt(signal) == pytest.approx(link["receiver_noise_rms_pt"] * 10 ** (link["reference_snr_db"] / 20))
    assert np.sqrt(noise * link["reference_noise_bandwidth_hz"]) == pytest.approx(link["receiver_noise_rms_pt"])


def test_noise_rms_scales_with_sqrt_bandwidth(cfg):
    channel = RelativeChannel(cfg)
    env = cfg["link"]["reference_environment"]
    narrow = channel.evaluate(Action(5, "BPSK", 0.5, 0.25), env, [])
    wide = channel.evaluate(Action(5, "BPSK", 0.5, 0.5), env, [])
    assert wide["noise_rms_pt"] / narrow["noise_rms_pt"] == pytest.approx(np.sqrt(2))
    assert wide["rx_field_rms_pt_assumed"] == pytest.approx(narrow["rx_field_rms_pt_assumed"])
    assert wide["snr_db"] == pytest.approx(narrow["snr_db"] - 10 * np.log10(2))


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "MSK"])
def test_esn0_uses_symbol_rate_and_noise_psd(cfg, modulation):
    channel = RelativeChannel(cfg)
    action = Action(7, modulation, 0.5, 0.5)
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


def test_relative_range_model_uses_declared_amplitude_exponent(cfg):
    close = copy.deepcopy(cfg["link"]["reference_environment"])
    distant = copy.deepcopy(close)
    distant["horizontal_distance_m"] *= 2
    ratio = np.abs(magnetic_transfer(5, distant, cfg["link"]) / magnetic_transfer(5, close, cfg["link"]))
    assert ratio == pytest.approx(0.5 ** cfg["link"]["range_spreading_exponent"])
    assert ratio < 1.0


def test_hard_occupancy_mask_accounts_for_waveform_width_and_guard(cfg):
    band = [{"low_hz": 5.9, "high_hz": 6.1, "mode": "hard"}]
    slow = Action(5.0, "BPSK", 0.5, 0.25)
    fast = Action(5.0, "BPSK", 0.5, 2 / 3)
    assert action_allowed(slow, band, cfg["link"])
    assert not action_allowed(fast, band, cfg["link"])
    link_no_guard = copy.deepcopy(cfg["link"])
    link_no_guard["guard_band_hz"] = 0.0
    assert action_allowed(fast, band, link_no_guard)


def test_soft_interference_integrates_only_overlap_bandwidth(cfg):
    channel = RelativeChannel(cfg)
    action = Action(5, "QPSK", 0.5, 0.5)
    env = cfg["link"]["reference_environment"]
    baseline = channel.evaluate(action, env, [])
    band = [{"low_hz": 4.95, "high_hz": 5.05, "mode": "soft", "interference_asd_pt_rt_hz": 0.3}]
    interfered = channel.evaluate(action, env, band)
    expected_psd_increase = 0.1 * 0.3 ** 2 / occupied_bandwidth(action, cfg["link"])
    assert interfered["effective_noise_psd_pt2_hz"] - baseline["effective_noise_psd_pt2_hz"] == pytest.approx(expected_psd_increase)
    assert interfered["legal"]
    assert interfered["esn0_db"] < baseline["esn0_db"]


def test_whole_band_occupied_blocks_every_action(cfg):
    bands = [{"low_hz": 3.0, "high_hz": 30.0, "mode": "hard"}]
    assert not any(action_allowed(action, bands, cfg["link"]) for action in build_actions(cfg))


def test_action_product_and_quick_config_are_independent(cfg):
    assert action_shape(cfg) == (13, 3, 4, 3)
    assert len(build_actions(cfg)) == 468
    fast = quick_config(cfg)
    assert fast["polar"]["n"] == 64
    assert cfg["polar"]["n"] == 256
    assert "SMOKE" in fast["run_label"]


@pytest.mark.parametrize("invalid", ["duplicate_axis", "not_power_two", "no_payload", "bad_frequency", "under_seabed"])
def test_configuration_rejects_invalid_inputs(cfg, invalid):
    if invalid == "duplicate_axis":
        cfg["actions"]["frequency_hz"].append(5)
    elif invalid == "not_power_two":
        cfg["polar"]["n"] = 192
    elif invalid == "no_payload":
        cfg["actions"]["code_rate"] = [0.01]
    elif invalid == "bad_frequency":
        cfg["actions"]["frequency_hz"] = [31]
    elif invalid == "under_seabed":
        cfg["environment"]["ranges"]["receiver_depth_m"] = [30, 150]
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("invalid", ["nan_rate", "infinite_noise", "reference_under_seabed", "negative_reference_noise", "absolute_zero_ice", "overwide_to_dc", "nonfinite_occupancy"])
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
        cfg["actions"]["symbol_rate"] = [10.0]
    elif invalid == "nonfinite_occupancy":
        cfg["environment"]["occupied_bands"] = [{"low_hz": 5.0, "high_hz": float("nan"), "mode": "hard"}]
    with pytest.raises(ValueError):
        validate_config(cfg)

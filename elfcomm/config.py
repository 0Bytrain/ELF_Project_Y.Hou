from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .codecs import RATES, SUPPORTED_SCHEMES, build_codec

ROOT = Path(__file__).resolve().parents[1]
ACTION_AXES = ("frequency_hz", "tx_power_kw", "waveform_mode", "coding_mode")
ENVIRONMENT_KEYS = (
    "ice_thickness_m", "ice_temperature_c", "seawater_conductivity_s_m",
    "receiver_depth_m", "horizontal_distance_m", "environment_noise_asd_pt_rt_hz",
)


@dataclass(frozen=True)
class Action:
    frequency_hz: float
    tx_power_kw: float
    payload_rate: float
    modulation: str
    symbol_rate: float
    code_rate: float
    coding_scheme: str = "Polar"

    @property
    def bits_per_symbol(self):
        return 2 if self.modulation == "QPSK" else 1

    @property
    def waveform_mode(self):
        return (self.modulation, self.symbol_rate)

    @property
    def coding_mode(self):
        return (self.coding_scheme, self.code_rate)

    def as_dict(self):
        return asdict(self)


def waveform_modes(cfg):
    """Fixed-order (modulation, symbol_rate) list; this is the waveform-mode axis."""
    return [tuple(item) for item in cfg["actions"]["waveform_mode"]]


def coding_modes(cfg):
    """Fixed-order (coding_scheme, nominal_code_rate) list; the coding axis."""
    return [tuple(item) for item in cfg["actions"]["coding_mode"]]


def coding_schemes(cfg):
    return list(dict.fromkeys(scheme for scheme, _ in coding_modes(cfg)))


def payload_rate_for(cfg, scheme, code_rate):
    return build_codec(scheme, n=cfg["polar"]["n"], rate=code_rate,
                       crc_bits=cfg["polar"]["crc_bits"],
                       list_size=cfg["polar"]["list_size"]).payload_rate


def payload_rates(cfg):
    """Effective payload rates in coding-axis order (derived, not an axis)."""
    return [payload_rate_for(cfg, scheme, rate) for scheme, rate in coding_modes(cfg)]


def waveform_mode_index(action, cfg):
    return waveform_modes(cfg).index((action.modulation, action.symbol_rate))


def coding_mode_index(action, cfg):
    return coding_modes(cfg).index((action.coding_scheme, action.code_rate))

#加载配置
def load_config(path=None):
    config_path = Path(path) if path else ROOT / "configs" / "default.json"
    with config_path.open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    return cfg

#留下有效配置，拦下坏配置
def validate_config(cfg):
    def check_finite(value, name="config"):
        if isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f"{name}.{key}")
        elif isinstance(value, list):
            for item in value:
                check_finite(item, name)
        elif isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Nonfinite configuration value: {name}")
    check_finite(cfg)
    if cfg.get("schema_version") != 2:
        raise ValueError("Unsupported configuration schema (expected 2; old schema 1 is rejected)")
    n = cfg["polar"]["n"]
    if not isinstance(n, int) or n < 32 or n & (n - 1):
        raise ValueError("Polar n must be a power of two >= 32")
    if cfg["polar"]["crc_bits"] != 11:
        raise ValueError("This version uses CRC-11")
    for axis in ("frequency_hz", "tx_power_kw"):
        values = cfg["actions"][axis]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate action axis: {axis}")
    if not set(cfg["actions"]["modulation"]) <= {"BPSK", "QPSK", "MSK"}:
        raise ValueError("Unsupported modulation")
    if any(x <= 0 for x in cfg["actions"]["symbol_rate"]) or any(x <= 0 for x in cfg["actions"]["tx_power_kw"]):
        raise ValueError("Symbol rates and TX powers must be positive")
    coding = cfg["actions"].get("coding_mode")
    if not isinstance(coding, list) or not coding or len(set(map(tuple, coding))) != len(coding):
        raise ValueError("Empty or duplicate coding-mode axis")
    for mode in coding:
        if not isinstance(mode, list) or len(mode) != 2:
            raise ValueError("Each coding mode needs [coding_scheme, nominal_code_rate]")
        scheme, rate = mode
        if scheme not in SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported coding scheme: {scheme}")
        if not any(abs(float(rate) - float(value)) <= 1e-12 for value in RATES[scheme]):
            raise ValueError(f"Unsupported {scheme} code rate: {rate}")
        if build_codec(scheme, n=n, rate=rate, crc_bits=cfg["polar"]["crc_bits"],
                       list_size=cfg["polar"]["list_size"]).payload_bits <= 0:
            raise ValueError(f"Coding mode leaves no payload: {mode}")
    modes = waveform_modes(cfg)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Empty or duplicate waveform modes")
    for mode in modes:
        if len(mode) != 2:
            raise ValueError("Each waveform mode needs (modulation, symbol_rate)")
        modulation, symbol_rate = mode
        if modulation not in cfg["actions"]["modulation"] or symbol_rate not in cfg["actions"]["symbol_rate"]:
            raise ValueError(f"Waveform mode outside the source axes: {mode}")
    link = cfg["link"]
    if link["calibration_mode"] != "relative_attenuation_anchor":
        raise ValueError("Only the relative attenuation-anchor calibration is supported")
    for forbidden in ("fixed_tx_power_kw", "range_spreading_exponent", "reference_snr_db"):
        if forbidden in link:
            raise ValueError(f"Legacy link field is not allowed: {forbidden}")
    propagation_model = link["propagation_model"]
    if propagation_model not in ("lu_attenuation", "lwpc_table"):
        raise ValueError("propagation_model must be 'lu_attenuation' or 'lwpc_table'")
    if propagation_model == "lu_attenuation":
        if link["beta_scale"] <= 0 or link["propagation_min_km"] <= 0:
            raise ValueError("beta_scale and propagation_min_km must be positive")
        if link["ionosphere_mode"] not in ("day", "night"):
            raise ValueError("ionosphere_mode must be 'day' or 'night'")
        ionosphere = link.get("ionosphere", {})
        required = ("H_km", "sigma0_km") + (("sigma1_km",) if link["ionosphere_mode"] == "day"
                                            else ("hE_km", "sigmaE_s_m"))
        for key in required:
            if float(ionosphere.get(key, 0)) <= 0:
                raise ValueError(f"Invalid ionosphere parameter: {key}")
    else:
        if link["range_model"] != "lwpc_30hz_table":
            raise ValueError("Only the LWPC 30 Hz range table is supported")
        if abs(float(link["reference_frequency_hz"]) - 30.0) > 1e-9:
            raise ValueError("The LWPC 30 Hz range table requires reference_frequency_hz=30")
    source_model = link.get("source_model", "anchor")
    if source_model not in ("anchor", "current_moment"):
        raise ValueError("source_model must be 'anchor' or 'current_moment'")
    antenna = link.get("antenna")
    if source_model == "current_moment" and not isinstance(antenna, dict):
        raise ValueError("current_moment source_model requires link.antenna")
    if antenna is not None:
        if not isinstance(antenna, dict):
            raise ValueError("link.antenna must be an object")
        if float(antenna.get("current_moment_a_m", 0)) <= 0:
            raise ValueError("antenna.current_moment_a_m must be positive")
        reference_antenna_frequency = float(
            antenna.get("current_moment_reference_frequency_hz", link["reference_frequency_hz"]))
        if reference_antenna_frequency <= 0:
            raise ValueError("antenna reference frequency must be positive")
        if antenna.get("unit_convention", "engineering") not in ("si", "engineering"):
            raise ValueError("antenna.unit_convention must be 'si' or 'engineering'")
        if float(antenna.get("pattern_factor", 1.0)) <= 0:
            raise ValueError("antenna.pattern_factor must be positive")
        length = antenna.get("effective_length_m")
        if length is not None and float(length) <= 0:
            raise ValueError("antenna.effective_length_m must be positive when provided")
        for key in ("antenna_site_conductivity_s_m", "path_ground_conductivity_s_m",
                    "ionosphere_conductivity_s_m", "earth_radius_km"):
            if float(antenna.get(key, 0)) <= 0:
                raise ValueError(f"Invalid antenna parameter: {key}")
        for key in ("input_resistance_ohm", "radiation_resistance_ohm"):
            rows = antenna.get(key)
            if not isinstance(rows, list) or len(rows) < 2:
                raise ValueError(f"antenna.{key} needs at least two knots")
            previous = -1.0
            for row in rows:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"antenna.{key} must contain [frequency_hz, ohm] pairs")
                frequency, value = float(row[0]), float(row[1])
                if frequency <= previous or frequency <= 0 or value <= 0:
                    raise ValueError(f"antenna.{key} must be strictly increasing and positive")
                previous = frequency
            frequencies = [float(row[0]) for row in rows]
            if not min(frequencies) <= reference_antenna_frequency <= max(frequencies):
                raise ValueError(f"antenna reference frequency is outside {key} knots")
    if link["reference_tx_power_kw"] <= 0 or link["reference_signal_rms_pt"] <= 0:
        raise ValueError("Reference TX power and reference signal must be positive")
    if link["field_anchor_distance_m"] <= 0:
        raise ValueError("Field anchor distance must be positive")
    if abs(link["field_anchor_distance_m"] - link["reference_environment"]["horizontal_distance_m"]) > 1.0:
        raise ValueError("field_anchor_distance_m must equal the reference point distance")
    for key in ("max_apparent_power_kva", "rated_active_power_kw", "power_factor"):
        if link[key] <= 0:
            raise ValueError(f"Expected positive hardware power metadata: {key}")
    if link["power_factor"] > 1:
        raise ValueError("power_factor must be in (0,1]")
    if link["reference_tx_power_kw"] not in cfg["actions"]["tx_power_kw"]:
        raise ValueError("reference_tx_power_kw must be one of actions.tx_power_kw")
    if max(cfg["actions"]["tx_power_kw"]) > link["rated_active_power_kw"] + 1e-9:
        raise ValueError("TX power axis exceeds the rated active power")
    if link["source_spectrum"] != "flat_field_assumption":
        raise ValueError("Supply a validated source provider before changing spectrum model")
    if link["receiver_noise_rms_pt"] <= 0 or link["reference_noise_bandwidth_hz"] <= 0:
        raise ValueError("Noise RMS and reference bandwidth must be positive")
    if not 0 <= link["rrc_rolloff"] <= 1 or link["guard_band_hz"] < 0:
        raise ValueError("Invalid bandwidth or guard parameters")
    fmin, fmax = link["allowed_band_hz"]
    if not 0 < fmin < fmax or any(not fmin <= f <= fmax for f in cfg["actions"]["frequency_hz"]):
        raise ValueError("Frequency outside configured operating band")
    for key in ENVIRONMENT_KEYS:
        lower, upper = cfg["environment"]["ranges"][key]
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"Invalid environment range: {key}")
        if not lower <= link["reference_environment"][key] <= upper:
            raise ValueError(f"Reference environment outside configured ranges: {key}")
    if propagation_model == "lu_attenuation":
        if cfg["environment"]["ranges"]["horizontal_distance_m"][0] < link["propagation_min_km"] * 1000:
            raise ValueError("Environment distance range starts below the propagation formula minimum")
    else:
        if len(link["range_table_valid_km"]) != 2 or not 0 < link["range_table_valid_km"][0] < link["range_table_valid_km"][1]:
            raise ValueError("range_table_valid_km must be [low, high]")
        if cfg["environment"]["ranges"]["horizontal_distance_m"][0] < link["range_table_valid_km"][0] * 1000:
            raise ValueError("Environment distance range starts before the LWPC table valid segment")
        if cfg["environment"]["ranges"]["horizontal_distance_m"][1] > link["range_table_valid_km"][1] * 1000:
            raise ValueError("Environment distance range ends after the LWPC table valid segment")
    if not link["allowed_band_hz"][0] <= link["reference_frequency_hz"] <= link["allowed_band_hz"][1]:
        raise ValueError("Reference frequency outside the allowed band")
    for key in ("ice_thickness_m", "seawater_conductivity_s_m", "receiver_depth_m", "horizontal_distance_m"):
        if cfg["environment"]["ranges"][key][0] <= 0 or link["reference_environment"][key] <= 0:
            raise ValueError(f"Nonpositive physical parameter: {key}")
    if cfg["environment"]["ranges"]["receiver_depth_m"][1] >= link["water_depth_m"]:
        raise ValueError("Receiver must be above seabed")
    if not 0 < link["reference_environment"]["receiver_depth_m"] < link["water_depth_m"]:
        raise ValueError("Reference receiver must be above seabed")
    if link["reference_environment"]["environment_noise_asd_pt_rt_hz"] < 0:
        raise ValueError("Reference noise ASD cannot be negative")
    if link["ice"]["relaxation_model"] != "arrhenius_pure_ice_approximation":
        raise ValueError("Unsupported ice relaxation model")
    if not -73.15 <= link["reference_environment"]["ice_temperature_c"] <= 0:
        raise ValueError("Reference ice temperature outside implemented approximation")
    tmin, tmax = cfg["environment"]["ranges"]["ice_temperature_c"]
    if tmin < -73.15 or tmax > 0:
        raise ValueError("Ice temperature outside implemented approximation")
    largest_factor = max(1 + link["rrc_rolloff"], link["msk_occupied_bandwidth_factor"])
    if min(cfg["actions"]["frequency_hz"]) <= max(cfg["actions"]["symbol_rate"]) * largest_factor / 2:
        raise ValueError("Candidate waveform extends to zero/negative physical frequency")
    for key in ("max_occupied_bandwidth_hz", "water_depth_m", "reference_tx_power_kw", "msk_occupied_bandwidth_factor"):
        if link[key] <= 0:
            raise ValueError(f"Expected positive link parameter: {key}")
    if cfg["environment"]["ranges"]["environment_noise_asd_pt_rt_hz"][0] < 0:
        raise ValueError("Noise ASD cannot be negative")
    for band in cfg["environment"]["occupied_bands"]:
        validate_band(band)
    knots = np.asarray(cfg["ber_table"]["esn0_db"], dtype=float)
    if len(knots) < 2 or not np.all(np.diff(knots) > 0):
        raise ValueError("Es/N0 knots must be strictly increasing")
    for section in ("ber_table", "validation"):
        if not 1 <= cfg[section]["min_frames"] <= cfg[section]["max_frames"]:
            raise ValueError(f"Invalid frame budget: {section}")
    if cfg["training"]["steps_per_episode"] <= 0 or cfg["training"]["episodes"] <= 0:
        raise ValueError("Training length must be positive")
    for key in ("evaluation_interval", "evaluation_episodes", "tensor_rank", "hidden_size", "batch_size"):
        if cfg["training"][key] <= 0:
            raise ValueError(f"Training setting must be positive: {key}")
    if not 0 < cfg["objective"]["target_ber"] < 0.5 or cfg["objective"]["target_goodput_bps"] <= 0:
        raise ValueError("Invalid BER or goodput target")
    if cfg["objective"].get("mode", "weighted") not in ("weighted", "ber_constrained_min_power"):
        raise ValueError("Unsupported objective mode")
    if cfg["objective"].get("preamble_symbols", 0) < 0:
        raise ValueError("preamble_symbols cannot be negative")
    training = cfg["training"]
    if training.get("exploration_schedule", "linear") not in ("linear", "phase_cosine"):
        raise ValueError("Unsupported exploration schedule")
    for key in ("epsilon_start", "epsilon_end"):
        if not np.isfinite(training[key]) or not 0 <= training[key] <= 1:
            raise ValueError(f"{key} must be finite and lie in [0, 1]")
    if not np.isfinite(training["epsilon_decay_fraction"]) or not 0 < training["epsilon_decay_fraction"] <= 1:
        raise ValueError("epsilon_decay_fraction must lie in (0, 1]")
    balance = training.get("replay_balance_fraction", 0.0)
    if not np.isfinite(balance) or not 0 <= balance < 1:
        raise ValueError("replay_balance_fraction must lie in [0, 1)")
    for key in ("additive_value_heads", "reset_replay_on_channel_shift"):
        if not isinstance(training.get(key, False), bool):
            raise ValueError(f"{key} must be boolean")
    exploration = training.get("structured_exploration", {})
    if not isinstance(exploration.get("enabled", False), bool):
        raise ValueError("structured_exploration.enabled must be boolean")
    if not np.isfinite(exploration.get("global_fraction", 0.5)) or not 0 <= exploration.get("global_fraction", 0.5) <= 1:
        raise ValueError("structured_exploration.global_fraction must lie in [0, 1]")
    radius = exploration.get("frequency_neighbor_radius", 3)
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("frequency_neighbor_radius must be a positive integer")
    after = (cfg.get("channel_shift") or {}).get("epsilon_after", training["epsilon_end"])
    if not np.isfinite(after) or not 0 <= after <= 1:
        raise ValueError("channel_shift.epsilon_after must lie in [0, 1]")
    window = training.get("decision_plot_window", 25)
    if not isinstance(window, int) or window < 1:
        raise ValueError("decision_plot_window must be a positive integer")
    lr = training["learning_rate"]
    if not 0 < training.get("learning_rate_final", lr) <= lr:
        raise ValueError("learning_rate_final must be positive and <= learning_rate")
    if not 0 < training.get("evaluation_ema_alpha", 0.2) <= 1:
        raise ValueError("evaluation_ema_alpha must lie in (0, 1]")
    control = cfg.get("decision", {})
    if not isinstance(control.get("ber_feasibility_filter", False), bool):
        raise ValueError("ber_feasibility_filter must be boolean")
    if not np.isfinite(control.get("same_power_goodput_gain_bps", 0.01)) or control.get("same_power_goodput_gain_bps", 0.01) < 0:
        raise ValueError("same_power_goodput_gain_bps must be finite and nonnegative")
    stability = training.get("convergence", {})
    if stability.get("window", 20) < 2 or stability.get("min_phase_episodes", 200) < 1:
        raise ValueError("Invalid convergence observation window")
    for key in ("power_tolerance_kw", "reward_tolerance", "max_configuration_switch_fraction"):
        if not np.isfinite(stability.get(key, 0)) or stability.get(key, 0) < 0:
            raise ValueError(f"Invalid convergence tolerance: {key}")
    if stability.get("max_configuration_switch_fraction", 0.05) > 1:
        raise ValueError("Configuration switch fraction must not exceed 1")
    if stability.get("candidate_lock_evaluations", 5) < 2:
        raise ValueError("Candidate locking requires at least two evaluations")
    if control.get("same_power_confirmation_rounds", 1) < 1:
        raise ValueError("same_power_confirmation_rounds must be positive")
    if not isinstance(training.get("dynamic_environment", True), bool):
        raise ValueError("dynamic_environment must be boolean")
    if not isinstance(training.get("ber_feasibility_filter", False), bool):
        raise ValueError("training.ber_feasibility_filter must be boolean")
    refinement = cfg["ber_table"].get("refinement", {})
    if refinement.get("enabled", False):
        if not 1 <= refinement["min_frames"] <= refinement["max_frames"]:
            raise ValueError("Invalid refined-table frame budget")
        if not isinstance(refinement["subdivisions"], int) or refinement["subdivisions"] < 1:
            raise ValueError("BER refinement subdivisions must be a positive integer")
        if not 0 < refinement["target_ber"] < 0.5 or refinement["near_target_multiplier"] < 1:
            raise ValueError("Invalid BER refinement target")
        if refinement["target_bit_errors"] < 1:
            raise ValueError("Refined-table target error count must be positive")
    validate_controlled_experiment(cfg)


def validate_controlled_experiment(cfg):
    experiment = cfg.get("controlled_experiment")
    if experiment is None:
        return
    if experiment["variable"] == "seawater_conductivity_s_m":
        raise ValueError("Conductivity-change experiments have been removed; keep conductivity fixed")
    before = cfg.get("fixed_environment", {})
    after = (cfg.get("channel_shift") or {}).get("environment", {})
    if set(before) != set(ENVIRONMENT_KEYS) or set(after) != set(ENVIRONMENT_KEYS):
        raise ValueError("Controlled experiment requires two complete environments")
    changed = {key for key in ENVIRONMENT_KEYS if before[key] != after[key]}
    if changed != {experiment["variable"]}:
        raise ValueError("Controlled experiment must change exactly its declared environment variable")
    if cfg["training"].get("dynamic_environment", True):
        raise ValueError("Controlled experiments must disable within-episode environment drift")
    if cfg["environment"].get("random_occupancy", True):
        raise ValueError("Controlled experiments must disable random occupancy")


def validate_band(band):
    if not np.isfinite([band["low_hz"], band["high_hz"], band.get("interference_asd_pt_rt_hz", 0)]).all():
        raise ValueError("Occupied-band values must be finite")
    if band["mode"] not in ("hard", "soft"):
        raise ValueError("Band mode must be hard or soft")
    if band["low_hz"] >= band["high_hz"]:
        raise ValueError("Occupied band needs low_hz < high_hz")
    if band["mode"] == "soft" and band.get("interference_asd_pt_rt_hz", -1) < 0:
        raise ValueError("Soft band requires nonnegative interference ASD")


def action_shape(cfg):
    return (len(cfg["actions"]["frequency_hz"]), len(cfg["actions"]["tx_power_kw"]),
            len(waveform_modes(cfg)), len(coding_modes(cfg)))


# 四重循环：频点 → 功率 → 波形模式（调制+符号率）→ 编码模式（编码方式+编码率）
def build_actions(cfg):
    actions = []
    for frequency in cfg["actions"]["frequency_hz"]:
        for power in cfg["actions"]["tx_power_kw"]:
            for modulation, symbol_rate in waveform_modes(cfg):
                for scheme, code_rate in coding_modes(cfg):
                    payload_rate = payload_rate_for(cfg, scheme, code_rate)
                    actions.append(Action(frequency, power, payload_rate, modulation,
                                          symbol_rate, code_rate, scheme))
    return actions


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_path(path):
    path = Path(path)
    # Keep relative artifact paths relative to the project working directory.
    # This avoids Windows/PyTorch failures when the project root contains
    # non-ASCII characters; the CLI is intended to run from this directory.
    return path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows: an antivirus/indexer can briefly lock the target file; retry the rename.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25)

#快速配置，缩减训练时长。
def quick_config(cfg):
    cfg = copy.deepcopy(cfg)    #深拷贝不影响原配置
    cfg["polar"].update(n=64, list_size=2)  #码长256--64
    cfg["actions"].update(frequency_hz=[17, 25, 33, 41, 55, 60, 100, 160, 220, 300],
                          tx_power_kw=[500.0, 62.5],
                          coding_mode=[["Polar", 1.0 / 3.0], ["Polar", 0.5],
                                       ["Polar", 2.0 / 3.0], ["Polar", 0.75],
                                       ["CC", 0.5], ["CC", 2.0 / 3.0], ["CC", 0.75]])
    cfg["objective"]["target_goodput_bps"] = 0.35
    cfg["ber_table"].update(esn0_db=[-6, 0, 4, 8, 14, 24, 30], min_frames=2, max_frames=4, target_bit_errors=30,
                            path="outputs/plan_demo/ber_table_cc.json")
    cfg["training"].update(episodes=25, steps_per_episode=12, batch_size=32, warmup_steps=32,
                           evaluation_interval=5, evaluation_episodes=3, hidden_size=64, tensor_rank=8) #500轮变为25轮
    cfg["validation"].update(min_frames=4, max_frames=16, target_bit_errors=50)
    cfg["output_dir"] = "outputs/plan_demo" # quick 统一写到 plan_demo，保持一个 quick 产物
    cfg["run_label"] = "SMOKE_ONLY_NOT_RELIABILITY_EVIDENCE"
    validate_config(cfg)    #改完配置之后依旧要过门禁，拦下非法配置
    return cfg

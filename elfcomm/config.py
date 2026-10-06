from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTION_AXES = ("frequency_hz", "modulation", "code_rate", "symbol_rate")
ENVIRONMENT_KEYS = (
    "ice_thickness_m", "ice_temperature_c", "seawater_conductivity_s_m",
    "receiver_depth_m", "horizontal_distance_m", "environment_noise_asd_pt_rt_hz",
)


@dataclass(frozen=True)
class Action:
    frequency_hz: float
    modulation: str
    code_rate: float
    symbol_rate: float

    @property
    def bits_per_symbol(self):
        return 2 if self.modulation == "QPSK" else 1

    def as_dict(self):
        return asdict(self)


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "configs" / "default.json"
    with config_path.open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    return cfg


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
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    n = cfg["polar"]["n"]
    if not isinstance(n, int) or n < 32 or n & (n - 1):
        raise ValueError("Polar n must be a power of two >= 32")
    if cfg["polar"]["crc_bits"] != 11:
        raise ValueError("This version uses CRC-11")
    for rate in cfg["actions"]["code_rate"]:
        if not 0 < rate < 1 or round(n * rate) <= 11:
            raise ValueError("Polar information block must contain payload plus CRC-11")
    for axis in ACTION_AXES:
        values = cfg["actions"][axis]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate action axis: {axis}")
    if not set(cfg["actions"]["modulation"]) <= {"BPSK", "QPSK", "MSK"}:
        raise ValueError("Unsupported modulation")
    if any(x <= 0 for x in cfg["actions"]["symbol_rate"]):
        raise ValueError("Symbol rates must be positive")
    link = cfg["link"]
    if link["calibration_mode"] != "relative":
        raise ValueError("Only explicit relative calibration is supported")
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
    for key in ("max_occupied_bandwidth_hz", "water_depth_m", "fixed_tx_power_kw", "msk_occupied_bandwidth_factor"):
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
    return tuple(len(cfg["actions"][key]) for key in ACTION_AXES)


def build_actions(cfg):
    return [Action(*values) for values in itertools.product(*(cfg["actions"][key] for key in ACTION_AXES))]


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
    temporary.replace(path)


def quick_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["polar"].update(n=64, list_size=2)
    cfg["ber_table"].update(esn0_db=[-6, 0, 4, 8, 14, 24, 30], min_frames=2, max_frames=4, target_bit_errors=30,
                            path="outputs/quick/ber_table.json")
    cfg["training"].update(episodes=25, steps_per_episode=12, batch_size=32, warmup_steps=32,
                           evaluation_interval=5, evaluation_episodes=3, hidden_size=64, tensor_rank=8)
    cfg["validation"].update(min_frames=4, max_frames=16, target_bit_errors=50)
    cfg["output_dir"] = "outputs/quick"
    cfg["run_label"] = "SMOKE_ONLY_NOT_RELIABILITY_EVIDENCE"
    validate_config(cfg)
    return cfg

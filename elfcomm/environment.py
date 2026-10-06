from __future__ import annotations

import copy

import numpy as np

from .channel import RelativeChannel
from .config import ENVIRONMENT_KEYS, action_shape, build_actions, validate_band
from .performance import evaluate_action, utility


class CommunicationEnvironment:
    def __init__(self, cfg, table, seed=None):
        self.cfg = cfg
        self.table = table
        self.actions = build_actions(cfg)
        self.shape = action_shape(cfg)
        self.channel = RelativeChannel(cfg)
        self.rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
        self.env = {}
        self.bands = []
        self.random_bands = []
        self.previous_action = None
        self.step_count = 0
        self.dynamic = True
        self.reset()

    @property
    def state_dim(self):
        return len(self.state())

    def _sample_environment(self):
        env = {}
        for key, bounds in self.cfg["environment"]["ranges"].items():
            low, high = bounds
            if key == "environment_noise_asd_pt_rt_hz" and low > 0:
                env[key] = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
            else:
                env[key] = float(self.rng.uniform(low, high))
        return env

    def _sample_bands(self):
        settings = self.cfg["environment"]
        if not settings["random_occupancy"]:
            return []
        bands = []
        for _ in range(self.rng.integers(settings["random_band_count_max"] + 1)):
            center = float(self.rng.choice(self.cfg["actions"]["frequency_hz"]))
            width = float(self.rng.uniform(*settings["random_band_width_hz"]))
            mode = "hard" if self.rng.random() < settings["random_hard_probability"] else "soft"
            band = {"low_hz": center - width / 2, "high_hz": center + width / 2, "mode": mode}
            if mode == "soft":
                band["interference_asd_pt_rt_hz"] = float(self.rng.uniform(*settings["random_interference_asd_pt_rt_hz"]))
            bands.append(band)
        return bands

    def reset(self, environment=None, occupied_bands=None, dynamic=True):
        self.dynamic = dynamic
        self.env = self._sample_environment() if environment is None else copy.deepcopy(environment)
        if set(self.env) != set(ENVIRONMENT_KEYS) or not all(np.isfinite(v) for v in self.env.values()):
            raise ValueError("Environment needs exactly the six finite configured physical parameters")
        if self.env["ice_thickness_m"] < 0 or not 0 < self.env["receiver_depth_m"] < self.cfg["link"]["water_depth_m"]:
            raise ValueError("Invalid ice thickness or receiver depth")
        if self.env["environment_noise_asd_pt_rt_hz"] < 0 or self.env["seawater_conductivity_s_m"] <= 0:
            raise ValueError("Noise ASD must be nonnegative and water conductivity positive")
        self.manual_bands = copy.deepcopy(self.cfg["environment"]["occupied_bands"] if occupied_bands is None else occupied_bands)
        for band in self.manual_bands:
            validate_band(band)
        self.random_bands = self._sample_bands() if dynamic and occupied_bands is None else []
        self.bands = self.manual_bands + self.random_bands
        self.previous_action = None
        self.step_count = 0
        self._refresh()
        return self.state(), self.mask.copy()

    def _refresh(self):
        self.metrics = self.channel.evaluate_many(self.actions, self.env, self.bands)
        self.mask = self.metrics["legal"].copy()

    def state(self):
        normalized = []
        for key in ENVIRONMENT_KEYS:
            low, high = self.cfg["environment"]["ranges"][key]
            normalized.append(2 * (self.env[key] - low) / max(high - low, 1e-12) - 1)
        frequency = self.channel.frequency_features(self.cfg["actions"]["frequency_hz"], self.env, self.bands)
        legal = self.mask.reshape(self.shape)[:, :, 0, :].ravel().astype(float)
        noise = self.metrics["effective_noise_psd_pt2_hz"].reshape(self.shape)[:, :, 0, :].ravel()
        noise = np.log10(noise / self.channel.sensor_psd) / 3
        previous = np.zeros(5)
        if self.previous_action is not None:
            previous[:4] = np.array(np.unravel_index(self.previous_action, self.shape)) / np.maximum(np.array(self.shape) - 1, 1)
            previous[4] = 1
        return np.concatenate([normalized, frequency, legal, noise, previous]).astype(np.float32)

    def action_metrics(self, index):
        values = {key: bool(value[index]) if key == "legal" else float(value[index]) for key, value in self.metrics.items()}
        return evaluate_action(self.actions[index], values, self.table, self.cfg)

    def _evolve(self):
        settings = self.cfg["environment"]
        for key, (low, high) in settings["ranges"].items():
            if key == "environment_noise_asd_pt_rt_hz":
                current = max(self.env[key], max(low, 1e-8))
                value = current * np.exp(self.rng.normal(0, settings["noise_step_log_std"]))
            else:
                value = self.env[key] + self.rng.normal(0, (high - low) * settings["step_drift_fraction"])
            self.env[key] = float(np.clip(value, low, high))
        if self.rng.random() < settings["occupancy_change_probability"]:
            self.random_bands = self._sample_bands()
        self.bands = self.manual_bands + self.random_bands

    def step(self, action_index):
        if self.step_count >= self.cfg["training"]["steps_per_episode"]:
            raise RuntimeError("Episode finished; reset before stepping")
        environment_before = copy.deepcopy(self.env)
        bands_before = copy.deepcopy(self.bands)
        if action_index == -1:
            if np.any(self.mask):
                raise ValueError("Hold is reserved for no legal frequency-bandwidth combination")
            reward = float(self.cfg["objective"]["outage_reward"])
            info = {"transmitted": False, "outage": True, "ber_estimate": None,
                    "goodput_bps_estimate": 0.0, "reliability_certified": False}
            self.previous_action = None
        else:
            if not 0 <= action_index < len(self.actions) or not self.mask[action_index]:
                raise ValueError("Action is masked by spectral occupancy or bandwidth")
            info = self.action_metrics(action_index)
            switched = self.previous_action is not None and action_index != self.previous_action
            reward = utility(info, self.cfg, switched)
            info.update(transmitted=True, outage=False)
            self.previous_action = action_index
        info.update(environment=environment_before, occupied_bands=bands_before, action_index=action_index, reward=reward)
        self.step_count += 1
        done = self.step_count >= self.cfg["training"]["steps_per_episode"]
        if self.dynamic:
            self._evolve()
        self._refresh()
        return self.state(), reward, done, self.mask.copy(), info

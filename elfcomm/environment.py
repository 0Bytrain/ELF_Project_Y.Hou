from __future__ import annotations

import copy

import numpy as np

from .channel import RelativeChannel
from .config import ENVIRONMENT_KEYS, action_shape, build_actions, validate_band, waveform_modes
from .performance import evaluate_action, utility


class CommunicationEnvironment:
    def __init__(self, cfg, table, seed=None, ber_filter=None):
        self.cfg = cfg
        self.table = table
        self.ber_filter_enabled = (cfg.get("decision", {}).get("ber_feasibility_filter", False)
                                   if ber_filter is None else bool(ber_filter))
        self.actions = build_actions(cfg)       #频点×功率×有效数据载荷率×波形模式
        self.shape = action_shape(cfg)
        self.waveform_modes = waveform_modes(cfg)
        self.channel = RelativeChannel(cfg)     #物理信道
        self.rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
        self.env = {}
        self.bands = []
        self.random_bands = []
        self.previous_action = None
        self.step_count = 0
        self.dynamic = True
        self.ber_baseline = None
        self.power_cap_kw = None
        groups = {}
        for index, action in enumerate(self.actions):
            groups.setdefault((action.coding_scheme, action.modulation, action.code_rate), []).append(index)
        self.ber_groups = {key: np.array(indices) for key, indices in groups.items()}
        self.action_powers = np.array([a.tx_power_kw for a in self.actions])
        self.reset()

    @property
    def state_dim(self):
        return len(self.state())
    #在六个环境参数里随机采样
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
        min_hz = float(settings.get("random_occupancy_min_hz", 0.0))
        eligible = [f for f in self.cfg["actions"]["frequency_hz"] if f >= min_hz]
        # The high band is always hard.  Only when soft occupancy is explicitly
        # enabled do we also allow low-band centers, so the default high-band
        # behavior stays unchanged while random_hard_probability and the soft
        # interference ASD are no longer dead knobs.
        if float(settings.get("random_hard_probability", 1.0)) < 1.0:
            eligible += [f for f in self.cfg["actions"]["frequency_hz"] if f < min_hz]
        if not eligible:
            return []
        bands = []
        count = int(self.rng.integers(1, settings["random_band_count_max"] + 1))
        for _ in range(count):
            center = float(self.rng.choice(eligible))
            width = float(self.rng.uniform(*settings["random_band_width_hz"]))
            # High-band random occupancy is a hard point to avoid; low-band
            # random bands keep the previous hard/soft behavior.
            if center >= min_hz:
                mode = "hard"
            else:
                mode = "hard" if self.rng.random() < settings["random_hard_probability"] else "soft"
            band = {"low_hz": center - width / 2, "high_hz": center + width / 2, "mode": mode}
            if mode == "soft":
                band["interference_asd_pt_rt_hz"] = float(self.rng.uniform(*settings["random_interference_asd_pt_rt_hz"]))
            bands.append(band)
        return bands

    def reset(self, environment=None, occupied_bands=None, dynamic=True):
        self.dynamic = dynamic
        self.env = self._sample_environment() if environment is None else copy.deepcopy(environment)    #若没给环境就用样本环境，如果给了环境就深度copy然后用
        # 检查 6 个键齐全、数值有限、冰厚/深度/电导率等合法；
        if set(self.env) != set(ENVIRONMENT_KEYS) or not all(np.isfinite(v) for v in self.env.values()):
            raise ValueError("Environment needs exactly the six finite configured physical parameters")
        if self.env["ice_thickness_m"] < 0 or not 0 < self.env["receiver_depth_m"] < self.cfg["link"]["water_depth_m"]:
            raise ValueError("Invalid ice thickness or receiver depth")
        if self.env["environment_noise_asd_pt_rt_hz"] < 0 or self.env["seawater_conductivity_s_m"] <= 0:
            raise ValueError("Noise ASD must be nonnegative and water conductivity positive")
        # Keep state normalization in the declared [-1, 1] domain.  Scenario
        # and CLI overrides must not silently push the DQN out of distribution.
        for key, (lower, upper) in self.cfg["environment"]["ranges"].items():
            if not lower <= self.env[key] <= upper:
                raise ValueError(
                    f"Environment {key}={self.env[key]} is outside configured range [{lower}, {upper}]")
        # occupied_bands = 手动频带；只有 dynamic and occupied_bands is None 时才额外随机生成占用；
        self.manual_bands = copy.deepcopy(self.cfg["environment"]["occupied_bands"] if occupied_bands is None else occupied_bands)
        for band in self.manual_bands:
            validate_band(band)
        self.random_bands = self._sample_bands() if dynamic and occupied_bands is None else []
        self.bands = self.manual_bands + self.random_bands
        #清空 previous_action、step_count，_refresh()；
        self.previous_action = None
        self.step_count = 0
        self._refresh()
        return self.state(), self.mask.copy()
    #计算每个动作的链路指标和每个动作是否合法
    def set_power_cap(self, cap_kw):
        """Temporarily mask power levels above a cap (None removes the cap).

        This models a staged transmit-power controller: after a channel change
        the operator starts at a lower power and only raises it after the BER
        has stabilised and remains outside the degradation target.
        """
        self.power_cap_kw = None if cap_kw is None else float(cap_kw)
        self._refresh()

    def _refresh(self):
        self.metrics = self.channel.evaluate_many(self.actions, self.env, self.bands)   # 一次算出当前动作空间的链路指标（SNR、带宽、合法标记等）
        self.mask = self.metrics["legal"].copy()        # 与动作空间同长的布尔数组（属性，不是方法！）：mask[i] = 第 i 个动作是否合法；
        if self.power_cap_kw is not None:
            powers = np.array([action.tx_power_kw for action in self.actions])
            self.mask &= powers <= self.power_cap_kw + 1e-12
        self.spectral_mask = self.mask.copy()
        self.ber_estimates = np.full(len(self.actions), np.nan)
        self._feasible_mask = None
        self.minimum_feasible_power = None
        if self.ber_filter_enabled:
            self.mask = self.estimated_feasible_mask()

    def estimated_feasible_mask(self):
        if self._feasible_mask is None:
            for key, indices in self.ber_groups.items():
                indices = indices[self.spectral_mask[indices]]
                if not indices.size:
                    continue
                snr = self.metrics["esn0_db"][indices]
                if hasattr(self.table, "lookup_ber_many"):
                    values = self.table.lookup_ber_many(*key, snr)
                else:
                    values = [self.table.lookup(*key, s)["ber_estimate"] for s in snr]
                self.ber_estimates[indices] = values
            self._feasible_mask = self.spectral_mask & (self.ber_estimates <= self.cfg["objective"]["target_ber"])
            self.minimum_feasible_power = (float(np.min(self.action_powers[self._feasible_mask]))
                                           if self._feasible_mask.any() else None)
        return self._feasible_mask.copy()
   
    def state_layout(self):
        nf = len(self.cfg["actions"]["frequency_hz"])
        nw = len(self.waveform_modes)
        return [
            ("environment", 6, (-1.0, 1.0)),
            ("frequency", 3 * nf, (-1.0, 1.0)),
            ("legal", nf * nw, (0.0, 1.0)),
            ("noise", nf * nw, (-1.0, 1.0)),
            ("previous", 5, (0.0, 1.0)),
        ]

    # 显式布局；合法性和噪声只依赖频点/波形模式，不展开功率与载荷率。
    def state(self):
        normalized = []
        for key in ENVIRONMENT_KEYS:
            low, high = self.cfg["environment"]["ranges"][key]
            normalized.append(2 * (self.env[key] - low) / max(high - low, 1e-12) - 1)
        frequencies = self.cfg["actions"]["frequency_hz"]
        frequency = self.channel.frequency_features(frequencies, self.env, self.bands)
        mask4 = self.spectral_mask.reshape(self.shape)             # (nf, npower, nwave, ncoding)
        noise4 = self.metrics["effective_noise_psd_pt2_hz"].reshape(self.shape)
        legal_grid = mask4.any(axis=(1, 3))
        noise_grid = noise4[:, 0, :, 0]                   # noise is independent of power/coding
        noise = np.clip(np.log10(noise_grid / self.channel.sensor_psd) / 3.05, -1.0, 1.0)
        previous = np.zeros(5)
        if self.previous_action is not None:
            previous[:4] = np.array(np.unravel_index(self.previous_action, self.shape)) / np.maximum(np.array(self.shape) - 1, 1)
            previous[4] = 1
        return np.concatenate([normalized, frequency, legal_grid.ravel(), noise.ravel(), previous]).astype(np.float32)

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
                raise ValueError("Hold is reserved for no feasible transmission configuration")
            reward = float(self.cfg["objective"]["outage_reward"])
            info = {"transmitted": False, "outage": True, "ber_estimate": None,
                    "goodput_bps_estimate": 0.0, "reliability_certified": False,
                    "switched": False, "power_changed": False,
                    "ber_baseline_used": self.ber_baseline}
            info["outage_reason"] = ("ber_target_unreachable" if self.spectral_mask.any()
                                      else "no_legal_spectral_action")
            self.previous_action = None
        else:
            if not 0 <= action_index < len(self.actions) or not self.mask[action_index]:
                raise ValueError("Action is masked by spectrum, bandwidth, power cap or estimated BER")
            previous_power = None if self.previous_action is None else self.actions[self.previous_action].tx_power_kw
            info = self.action_metrics(action_index)
            switched = self.previous_action is not None and action_index != self.previous_action
            power_changed = previous_power is not None and info["tx_power_kw"] != previous_power
            reward = utility(info, self.cfg, switched, previous_power=previous_power, baseline_ber=self.ber_baseline)
            info.update(transmitted=True, outage=False, switched=switched, power_changed=power_changed,
                        ber_baseline_used=self.ber_baseline)
            if self.minimum_feasible_power is not None:
                minimum = self.minimum_feasible_power
                info.update(minimum_feasible_tx_power_kw=minimum,
                            power_optimality_gap_kw=info["tx_power_kw"] - minimum)
            self.previous_action = action_index
        info.update(environment=environment_before, occupied_bands=bands_before, action_index=action_index, reward=reward)
        self.step_count += 1
        done = self.step_count >= self.cfg["training"]["steps_per_episode"]
        if self.dynamic:
            self._evolve()
            self._refresh()
        return self.state(), reward, done, self.mask.copy(), info

from __future__ import annotations

import numpy as np

from .antenna import AntennaModel
from .propagation import PropagationModel
from .range_model import RangeModel

# NumPy 2.0 removed np.trapz (renamed to np.trapezoid); keep 1.x and 2.x both working.
try:
    _trapz = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapz = np.trapz

EPS0 = 8.8541878128e-12
MU0 = 4 * np.pi * 1e-7
KB_EV = 8.617333262e-5

#输入频率、相对介电常数、电导率；输出衰减和阻抗
def material(frequency_hz, relative_permittivity, conductivity_s_m=0.0):
    omega = 2 * np.pi * np.asarray(frequency_hz, dtype=float)
    if np.any(omega <= 0):
        raise ValueError("Frequency must be positive")
    epsilon = EPS0 * relative_permittivity - 1j * conductivity_s_m / omega
    gamma = np.sqrt(-omega ** 2 * MU0 * epsilon + 0j)
    gamma = np.where(gamma.real < 0, -gamma, gamma)
    impedance = 1j * omega * MU0 / gamma
    return gamma, impedance

#最后返回一个 H_local(f,env)（局部相对传输，不含距离）；后面只用模平方，相位不影响该指标。
#用到了冰厚、冰温、海水电导率、接收深度四个参数；水平距离由 RangeModel 单独处理。
def magnetic_transfer(frequency_hz, env, link):
    frequencies = np.asarray(frequency_hz, dtype=float)
    omega = 2 * np.pi * frequencies
    ice = link["ice"]
    temperature_k = env["ice_temperature_c"] + 273.15   #温度
    if temperature_k <= 0 or env["ice_thickness_m"] < 0 or env["receiver_depth_m"] < 0:
        raise ValueError("Invalid temperature, ice thickness or receiver depth")
    tau = ice["relaxation_prefactor_s"] * np.exp(ice["relaxation_activation_ev"] / (KB_EV * temperature_k))
    #冰的介电常数
    eps_ice = ice["epsilon_infinite"] + (ice["epsilon_static"] - ice["epsilon_infinite"]) / (1 + 1j * omega * tau)
    #三种介质介电常数和电导率
    gamma_air, z_air = material(frequencies, 1.0)
    gamma_ice, z_ice = material(frequencies, eps_ice, ice["conductivity_s_m"])
    gamma_water, z_water = material(frequencies, link["seawater_relative_permittivity"], env["seawater_conductivity_s_m"])
    #界面反射/透射
    r01 = (z_ice - z_air) / (z_ice + z_air)
    r12 = (z_water - z_ice) / (z_water + z_ice)
    t01 = 2 * z_ice / (z_air + z_ice)
    t12 = 2 * z_water / (z_ice + z_water)
    #冰层内
    ice_one_way = np.exp(-gamma_ice * env["ice_thickness_m"])
    #电场---磁场。把电场转化为磁场
    electric_transmission = t01 * t12 * ice_one_way / (1 + r01 * r12 * ice_one_way ** 2)
    magnetic_transmission = electric_transmission * z_air / z_water
    #海水深度衰减；距离由 RangeModel 单独处理，不进入 H_local
    water_one_way = np.exp(-gamma_water * env["receiver_depth_m"])
    return magnetic_transmission * water_one_way

#占用带宽：MSK：占用带宽 = 1.5 × 符号率；BPSK/QPSK：`1 + rrc_rolloff = 1.35` × 符号率
#**只依赖调制和符号率，不依赖码率**——这是 state 只取 117 切片的根源
def occupied_bandwidth(action, link):
    factor = link["msk_occupied_bandwidth_factor"] if action.modulation == "MSK" else 1 + link["rrc_rolloff"]
    return float(factor * action.symbol_rate)


def band_overlap(low, high, band_low, band_high):
    return max(0.0, min(high, band_high) - max(low, band_low))

#三条门禁：带宽不能超过 `max_occupied_bandwidth_hz`；
#频带 `[low, high]` 必须在允许频段 `[3, 30]` 内；
#不能和**硬占用**频带（含保护带 0.5 Hz）重叠；**软占用**不禁止，只加干扰
def action_allowed(action, occupied_bands, link):
    bandwidth = occupied_bandwidth(action, link)
    low, high = action.frequency_hz - bandwidth / 2, action.frequency_hz + bandwidth / 2
    if bandwidth > link["max_occupied_bandwidth_hz"] + 1e-12:
        return False
    if low < link["allowed_band_hz"][0] or high > link["allowed_band_hz"][1]:
        return False
    guard = link["guard_band_hz"]
    for band in occupied_bands:
        if band["mode"] == "hard" and band_overlap(low, high, band["low_hz"] - guard, band["high_hz"] + guard) > 1e-12:
            return False
    return True

#相对标定，所有的信号以参考点为锚，算的是|H(f,env)/H_ref|²` 的比值。所以模型不需要绝对功率
class RelativeChannel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.link = cfg["link"]
        link = self.link
        self.source_model = link.get("source_model", "anchor")
        if self.source_model not in ("anchor", "current_moment"):
            raise ValueError(f"Unsupported source model: {self.source_model}")
        self.antenna = AntennaModel(link) if self.source_model == "current_moment" else None
        self.propagation_model = link["propagation_model"]
        if self.propagation_model == "lu_attenuation":
            self.propagation = PropagationModel(link)
            self.range_model = None
        elif self.propagation_model == "lwpc_table":
            self.propagation = None
            self.range_model = RangeModel(link)
        else:
            raise ValueError(f"Unsupported propagation model: {self.propagation_model}")
        #参考频率/参考环境下的局部传输 H_local（不含距离）
        self.reference_transfer = complex(magnetic_transfer(link["reference_frequency_hz"], link["reference_environment"], link))
        #参考噪声带宽1hz，0.1**2/1；
        self.sensor_psd = link["receiver_noise_rms_pt"] ** 2 / link["reference_noise_bandwidth_hz"]
        ref_env = link["reference_environment"]["environment_noise_asd_pt_rt_hz"]
        #传感器psd+环境噪声ASD**2(功率相加)
        self.reference_noise_psd = self.sensor_psd + ref_env ** 2
        if self.source_model == "anchor":
            # 参考信号由锚点链派生，配置中的 reference_signal_rms_pt 是单一真源
            self.reference_signal_rms_pt = float(link["reference_signal_rms_pt"])
            if self.reference_signal_rms_pt <= 0:
                raise ValueError("reference_signal_rms_pt must be positive")
            # 锚点自检：E_anchor -> B_air -> H_local -> 参考距离处应等于配置值
            c = 299792458.0
            self.source_field_rms_pt = 10 ** (link["field_anchor_e_db_uv_m"] / 20) * 1e-6 / c * 1e12
            anchor_rms_pt = (self.source_field_rms_pt * abs(self.reference_transfer)
                             * 10 ** (self._distance_field_gain_db(link["reference_frequency_hz"],
                                                                  link["field_anchor_distance_m"]) / 20))
            if abs(anchor_rms_pt / self.reference_signal_rms_pt - 1) > 0.01:
                raise ValueError(
                    f"Reference signal does not match the source-field anchor chain: "
                    f"config={self.reference_signal_rms_pt:.4f} pT, anchor={anchor_rms_pt:.4f} pT")
        else:
            # 可选电流矩源模型：参考场由式(2.1.80)在参考频点/距离/功率算出，
            # 再乘局部冰水传输；配置里的 reference_signal_rms_pt 不参与。
            self.source_field_rms_pt = None
            self.reference_signal_rms_pt = float(
                self.antenna.field_rms_pt(link["reference_frequency_hz"],
                                          link["reference_environment"]["horizontal_distance_m"],
                                          link["reference_tx_power_kw"])
                * abs(self.reference_transfer))
            if self.reference_signal_rms_pt <= 0:
                raise ValueError("antenna source model produced a nonpositive reference field")
        self.reference_signal_mean_square = self.reference_signal_rms_pt ** 2
        self.reference_snr_db = float(10 * np.log10(
            self.reference_signal_mean_square
            / (self.reference_noise_psd * link["reference_noise_bandwidth_hz"])))

    def _distance_field_gain_db(self, frequency_hz, distance_m):
        if self.source_model == "current_moment":
            reference = self.antenna.reference_field_uv_m()
            field = self.antenna.field_uv_m(frequency_hz, distance_m,
                                           self.link["reference_tx_power_kw"])
            return 20 * np.log10(field / reference)
        if self.propagation is not None:
            return self.propagation.gain_db(frequency_hz, distance_m)
        return self.range_model.gain_db(distance_m)

    def distance_gain_db(self, frequency_hz, distance_m):
        return self._distance_field_gain_db(frequency_hz, distance_m)

    def _distance_power_ratio(self, frequency_hz, distance_m):
        return 10 ** (self._distance_field_gain_db(frequency_hz, distance_m) / 10)

    # 初始电场 E(P) ∝ sqrt(P)：先缩放源场，再经局部冰水 H_local 和波导传播
    def received_field_rms_pt(self, frequency_hz, env, tx_power_kw=None):
        local_field = np.abs(magnetic_transfer(frequency_hz, env, self.link))
        power = self.link["reference_tx_power_kw"] if tx_power_kw is None else tx_power_kw
        if self.source_model == "current_moment":
            # 式(2.1.80)已含源强、距离和波导衰减；这里只再乘局部冰水传输。
            air_field = self.antenna.field_rms_pt(frequency_hz, env["horizontal_distance_m"], power)
            return air_field * local_field
        propagation_field = 10 ** (self._distance_field_gain_db(frequency_hz, env["horizontal_distance_m"]) / 20)
        source_ratio = np.sqrt(power / self.link["reference_tx_power_kw"])
        return self.source_field_rms_pt * source_ratio * local_field * propagation_field

    # 信号功率 = 接收电场平方。功率已经在 received_field_rms_pt 里按
    # sqrt(P/P_ref) 进入，平方后正好是 P/P_ref；调用方不要再乘一次 P。
    def signal_mean_square(self, frequency_hz, env, tx_power_kw=None):
        return self.received_field_rms_pt(frequency_hz, env, tx_power_kw) ** 2
    
    #传感器本地+环境噪声*(f/f_ref)^(-1)`（配置里 `noise_spectrum_exponent=1`，两个噪声是按功率相加的
    def background_noise_psd(self, frequency_hz, env):
        shape = (np.asarray(frequency_hz, dtype=float) / self.link["reference_frequency_hz"]) ** (-self.link["noise_spectrum_exponent"])
        return self.sensor_psd + env["environment_noise_asd_pt_rt_hz"] ** 2 * shape
   
    #单个动作的指标
    def evaluate(self, action, env, occupied_bands):
        bandwidth = occupied_bandwidth(action, self.link)
        low = action.frequency_hz - bandwidth / 2
        high = action.frequency_hz + bandwidth / 2
        #`np.linspace(low, high, 17)`：在占用频带内取 17 个采样点，用来积分噪声和看波动。
        nodes = np.linspace(low, high, 17)
        signal = float(self.signal_mean_square(action.frequency_hz, env, action.tx_power_kw))
         #**背景噪声**：在 17 个点上算 PSD，积分后除以带宽 → 平均 PSD。其中_trapz(y, x)`：梯形积分
        background = float(_trapz(self.background_noise_psd(nodes, env), nodes) / bandwidth)
        #**软干扰**：interference / bandwidth：重叠带宽 × 干扰 ASD²，累加后除以带宽 → 平均干扰 PSD。
        interference = sum(band_overlap(low, high, b["low_hz"], b["high_hz"]) * b.get("interference_asd_pt_rt_hz", 0.0) ** 2
                           for b in occupied_bands if b["mode"] == "soft")
        #功率谱相加
        effective_psd = background + interference / bandwidth
        #噪声取整个占用带宽。
        noise_mean_square = effective_psd * bandwidth
        snr_db = 10 * np.log10(signal / noise_mean_square)
        #每个符号的能量 vs 噪声谱密度。注意分母是**符号率**，不是带宽。同一个信号下，符号率越低，Es/N0 越高（每个符号持续时间更长）。
        esn0_db = 10 * np.log10(signal / (effective_psd * action.symbol_rate))
        band_signal = self.signal_mean_square(nodes, env)
        #频带内信号最大/最小值的 dB 差，衡量信道在带内是否平坦。
        ripple_db = float(10 * np.log10(np.max(band_signal) / np.min(band_signal)))
        return {
            "rx_field_rms_pt_assumed": float(np.sqrt(signal)),
            "noise_rms_pt": float(np.sqrt(noise_mean_square)),
            "effective_noise_psd_pt2_hz": float(effective_psd),
            "snr_db": float(snr_db), "esn0_db": float(esn0_db),
            "occupied_bandwidth_hz": bandwidth,
            "channel_band_ripple_db": ripple_db,
            "tx_power_kw": action.tx_power_kw, "payload_rate": action.payload_rate,
            "waveform_mode": list(action.waveform_mode),
            "legal": action_allowed(action, occupied_bands, self.link),
            "receiver_model": "ideal_coherent_flat_equivalent_awgn",
        }

    def frequency_features(self, frequencies, env, occupied_bands):
        frequencies = np.asarray(frequencies, dtype=float)
        signal = self.signal_mean_square(frequencies, env)
        noise_psd = self.background_noise_psd(frequencies, env)
        reference_snr = 10 * np.log10(signal / (noise_psd * self.link["reference_noise_bandwidth_hz"]))
        hard, soft = [], []
        guard = self.link["guard_band_hz"]
        for frequency in frequencies:
            hard.append(float(any(b["mode"] == "hard" and b["low_hz"] - guard <= frequency <= b["high_hz"] + guard for b in occupied_bands)))
            power = sum(b.get("interference_asd_pt_rt_hz", 0) ** 2 for b in occupied_bands
                        if b["mode"] == "soft" and b["low_hz"] <= frequency <= b["high_hz"])
            soft.append(np.log1p(power / self.sensor_psd) / 10)
        return np.concatenate([reference_snr / 60, hard, soft]).astype(np.float32)  # 特征归一化上限 60 dB

    #动作先去重（code_rate 不影响物理），再展开
    def evaluate_many(self, actions, env, occupied_bands):
        keys = [(a.frequency_hz, a.modulation, a.symbol_rate, a.tx_power_kw) for a in actions]
        index = {}
        inverse = np.empty(len(actions), dtype=int)
        unique_actions = []
        for i, key in enumerate(keys):
            j = index.get(key)
            if j is None:
                j = len(index)
                index[key] = j
                unique_actions.append(actions[i])
            inverse[i] = j
        metrics = self._evaluate_vector(unique_actions, env, occupied_bands)
        return {key: (value[inverse] if isinstance(value, np.ndarray) else value) for key, value in metrics.items()}

    def _evaluate_vector(self, actions, env, occupied_bands):
        frequencies = np.array([a.frequency_hz for a in actions])
        rates = np.array([a.symbol_rate for a in actions])
        powers = np.array([a.tx_power_kw for a in actions])
        bandwidth = np.array([occupied_bandwidth(a, self.link) for a in actions])
        low, high = frequencies - bandwidth / 2, frequencies + bandwidth / 2
        nodes = frequencies[:, None] + bandwidth[:, None] * np.linspace(-0.5, 0.5, 17)
        signal = self.signal_mean_square(frequencies, env)
        power_ratio = powers / self.link["reference_tx_power_kw"]
        signal = signal * power_ratio
        background = _trapz(self.background_noise_psd(nodes, env), nodes, axis=1) / bandwidth
        interference = np.zeros(len(actions))
        legal = ((bandwidth <= self.link["max_occupied_bandwidth_hz"] + 1e-12)
                 & (low >= self.link["allowed_band_hz"][0]) & (high <= self.link["allowed_band_hz"][1]))
        for band in occupied_bands:
            if band["mode"] == "hard":
                overlap = np.minimum(high, band["high_hz"] + self.link["guard_band_hz"]) - np.maximum(low, band["low_hz"] - self.link["guard_band_hz"])
                legal &= overlap <= 1e-12
            else:
                overlap = np.maximum(0, np.minimum(high, band["high_hz"]) - np.maximum(low, band["low_hz"]))
                interference += overlap * band.get("interference_asd_pt_rt_hz", 0.0) ** 2
        psd = background + interference / bandwidth
        band_signal = self.signal_mean_square(nodes, env)
        ripple = 10 * np.log10(band_signal.max(axis=1) / band_signal.min(axis=1))
        return {
            "rx_field_rms_pt_assumed": np.sqrt(signal), "noise_rms_pt": np.sqrt(psd * bandwidth),
            "effective_noise_psd_pt2_hz": psd, "snr_db": 10 * np.log10(signal / (psd * bandwidth)),
            "esn0_db": 10 * np.log10(signal / (psd * rates)), "occupied_bandwidth_hz": bandwidth,
            "channel_band_ripple_db": ripple, "legal": legal,
            "tx_power_kw": powers,
        }

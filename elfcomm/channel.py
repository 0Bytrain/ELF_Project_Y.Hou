from __future__ import annotations

import numpy as np

# NumPy 2.0 removed np.trapz (renamed to np.trapezoid); keep 1.x and 2.x both working.
try:
    _trapz = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapz = np.trapz

EPS0 = 8.8541878128e-12
MU0 = 4 * np.pi * 1e-7
KB_EV = 8.617333262e-5


def material(frequency_hz, relative_permittivity, conductivity_s_m=0.0):
    omega = 2 * np.pi * np.asarray(frequency_hz, dtype=float)
    if np.any(omega <= 0):
        raise ValueError("Frequency must be positive")
    epsilon = EPS0 * relative_permittivity - 1j * conductivity_s_m / omega
    gamma = np.sqrt(-omega ** 2 * MU0 * epsilon + 0j)
    gamma = np.where(gamma.real < 0, -gamma, gamma)
    impedance = 1j * omega * MU0 / gamma
    return gamma, impedance


def magnetic_transfer(frequency_hz, env, link):
    frequencies = np.asarray(frequency_hz, dtype=float)
    omega = 2 * np.pi * frequencies
    ice = link["ice"]
    temperature_k = env["ice_temperature_c"] + 273.15
    if temperature_k <= 0 or env["ice_thickness_m"] < 0 or env["receiver_depth_m"] < 0:
        raise ValueError("Invalid temperature, ice thickness or receiver depth")
    tau = ice["relaxation_prefactor_s"] * np.exp(ice["relaxation_activation_ev"] / (KB_EV * temperature_k))
    eps_ice = ice["epsilon_infinite"] + (ice["epsilon_static"] - ice["epsilon_infinite"]) / (1 + 1j * omega * tau)
    gamma_air, z_air = material(frequencies, 1.0)
    gamma_ice, z_ice = material(frequencies, eps_ice, ice["conductivity_s_m"])
    gamma_water, z_water = material(frequencies, link["seawater_relative_permittivity"], env["seawater_conductivity_s_m"])
    r01 = (z_ice - z_air) / (z_ice + z_air)
    r12 = (z_water - z_ice) / (z_water + z_ice)
    t01 = 2 * z_ice / (z_air + z_ice)
    t12 = 2 * z_water / (z_ice + z_water)
    ice_one_way = np.exp(-gamma_ice * env["ice_thickness_m"])
    electric_transmission = t01 * t12 * ice_one_way / (1 + r01 * r12 * ice_one_way ** 2)
    magnetic_transmission = electric_transmission * z_air / z_water
    water_one_way = np.exp(-gamma_water * env["receiver_depth_m"])
    distance = env["horizontal_distance_m"]
    if distance <= 0:
        raise ValueError("Horizontal distance must be positive for the relative range model")
    range_ratio = (link["reference_environment"]["horizontal_distance_m"] / distance) ** link["range_spreading_exponent"]
    transfer = magnetic_transmission * water_one_way * range_ratio
    return transfer


def occupied_bandwidth(action, link):
    factor = link["msk_occupied_bandwidth_factor"] if action.modulation == "MSK" else 1 + link["rrc_rolloff"]
    return float(factor * action.symbol_rate)


def band_overlap(low, high, band_low, band_high):
    return max(0.0, min(high, band_high) - max(low, band_low))


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


class RelativeChannel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.link = cfg["link"]
        link = self.link
        self.reference_transfer = complex(magnetic_transfer(link["reference_frequency_hz"], link["reference_environment"], link))
        self.sensor_psd = link["receiver_noise_rms_pt"] ** 2 / link["reference_noise_bandwidth_hz"]
        ref_env = link["reference_environment"]["environment_noise_asd_pt_rt_hz"]
        self.reference_noise_psd = self.sensor_psd + ref_env ** 2
        self.reference_signal_mean_square = 10 ** (link["reference_snr_db"] / 10) * self.reference_noise_psd * link["reference_noise_bandwidth_hz"]

    def signal_mean_square(self, frequency_hz, env):
        ratio = np.abs(magnetic_transfer(frequency_hz, env, self.link) / self.reference_transfer) ** 2
        return self.reference_signal_mean_square * ratio

    def background_noise_psd(self, frequency_hz, env):
        shape = (np.asarray(frequency_hz, dtype=float) / self.link["reference_frequency_hz"]) ** (-self.link["noise_spectrum_exponent"])
        return self.sensor_psd + env["environment_noise_asd_pt_rt_hz"] ** 2 * shape

    def evaluate(self, action, env, occupied_bands):
        bandwidth = occupied_bandwidth(action, self.link)
        low = action.frequency_hz - bandwidth / 2
        high = action.frequency_hz + bandwidth / 2
        nodes = np.linspace(low, high, 17)
        signal = float(self.signal_mean_square(action.frequency_hz, env))
        background = float(_trapz(self.background_noise_psd(nodes, env), nodes) / bandwidth)
        interference = sum(band_overlap(low, high, b["low_hz"], b["high_hz"]) * b.get("interference_asd_pt_rt_hz", 0.0) ** 2
                           for b in occupied_bands if b["mode"] == "soft")
        effective_psd = background + interference / bandwidth
        noise_mean_square = effective_psd * bandwidth
        snr_db = 10 * np.log10(signal / noise_mean_square)
        esn0_db = 10 * np.log10(signal / (effective_psd * action.symbol_rate))
        band_signal = self.signal_mean_square(nodes, env)
        ripple_db = float(10 * np.log10(np.max(band_signal) / np.min(band_signal)))
        return {
            "rx_field_rms_pt_assumed": float(np.sqrt(signal)),
            "noise_rms_pt": float(np.sqrt(noise_mean_square)),
            "effective_noise_psd_pt2_hz": float(effective_psd),
            "snr_db": float(snr_db), "esn0_db": float(esn0_db),
            "occupied_bandwidth_hz": bandwidth,
            "channel_band_ripple_db": ripple_db,
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
        return np.concatenate([reference_snr / 30, hard, soft]).astype(np.float32)

    def evaluate_many(self, actions, env, occupied_bands):
        frequencies = np.array([a.frequency_hz for a in actions])
        rates = np.array([a.symbol_rate for a in actions])
        bandwidth = np.array([occupied_bandwidth(a, self.link) for a in actions])
        low, high = frequencies - bandwidth / 2, frequencies + bandwidth / 2
        nodes = frequencies[:, None] + bandwidth[:, None] * np.linspace(-0.5, 0.5, 17)
        signal = self.signal_mean_square(frequencies, env)
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
                interference += overlap * band["interference_asd_pt_rt_hz"] ** 2
        psd = background + interference / bandwidth
        band_signal = self.signal_mean_square(nodes, env)
        ripple = 10 * np.log10(band_signal.max(axis=1) / band_signal.min(axis=1))
        return {
            "rx_field_rms_pt_assumed": np.sqrt(signal), "noise_rms_pt": np.sqrt(psd * bandwidth),
            "effective_noise_psd_pt2_hz": psd, "snr_db": 10 * np.log10(signal / (psd * bandwidth)),
            "esn0_db": 10 * np.log10(signal / (psd * rates)), "occupied_bandwidth_hz": bandwidth,
            "channel_band_ripple_db": ripple, "legal": legal,
        }

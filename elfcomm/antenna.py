from __future__ import annotations

import hashlib
import json
import math

import numpy as np

C_LIGHT = 299792458.0
EPS0 = 8.8541878128e-12


class AntennaModel:
    """Horizontal low-profile ELF antenna source model (optional).

    The model follows the engineering chain in Liu Yong et al. (2021) and
    Lu Jianxun's textbook:

        I_a(f, P) = sqrt(P / R_in(f))                  (2.1.75)
        M(f, P)   = I_a(f, P) * L_eff                 (2.1.79)
        E_z(f, d) = sqrt(60) * pi * M / [h * lambda
                     * sqrt(sigma * a_e * sin(d / a_e))]
                     * cos(phi) * exp(-beta * d)       (2.1.80)
        beta      = 1/(2*sqrt(2)*h)
                     * (sqrt(omega*eps0/sigma_e)
                        + sqrt(omega*eps0/sigma_i))

    ``R_in`` and ``R_r`` are interpolated log-log from the 100 km antenna
    table in Lu's book (printed page 41).  ``L_eff`` is fixed so that the
    configured current moment is reproduced at the reference frequency and
    reference power; at other frequencies it follows I_a through R_in(f).

    Equation (2.1.80) has two common unit conventions in the source material.
    ``unit_convention='si'`` uses metres for h, lambda and a_e and returns
    V/m before conversion to uV/m; ``'engineering'`` evaluates the same
    formula with km for those lengths and then applies the explicit
    ``1/sqrt(1000)`` conversion needed to return the same physical uV/m.
    Both branches therefore agree; the optional report prints both.
    """

    def __init__(self, link):
        antenna = link["antenna"]
        self.link = link
        self.reference_frequency_hz = float(link["reference_frequency_hz"])
        self.reference_power_kw = float(link["reference_tx_power_kw"])
        self.reference_distance_m = float(link["reference_environment"]["horizontal_distance_m"])
        self.current_moment_ref_a_m = float(antenna["current_moment_a_m"])
        self.current_moment_reference_frequency_hz = float(
            antenna.get("current_moment_reference_frequency_hz", self.reference_frequency_hz))
        self.unit_convention = antenna.get("unit_convention", "engineering")
        if self.unit_convention not in ("si", "engineering"):
            raise ValueError("antenna.unit_convention must be 'si' or 'engineering'")
        self.pattern_factor = float(antenna.get("pattern_factor", 1.0))
        if self.pattern_factor < 0:
            raise ValueError("antenna.pattern_factor must be nonnegative")
        self.site_conductivity = float(antenna["antenna_site_conductivity_s_m"])
        self.path_conductivity = float(antenna["path_ground_conductivity_s_m"])
        self.ionosphere_conductivity = float(antenna["ionosphere_conductivity_s_m"])
        self.earth_radius_km = float(antenna["earth_radius_km"])
        self.h_km = float(link["ionosphere"]["H_km"])
        for name, value in (("site", self.site_conductivity), ("path", self.path_conductivity),
                            ("ionosphere", self.ionosphere_conductivity),
                            ("earth_radius", self.earth_radius_km), ("h", self.h_km)):
            if value <= 0:
                raise ValueError(f"Invalid antenna {name} parameter")
        if self.current_moment_ref_a_m <= 0:
            raise ValueError("antenna.current_moment_a_m must be positive")

        self.input_resistance = self._read_table(antenna["input_resistance_ohm"], "input_resistance_ohm")
        self.radiation_resistance = self._read_table(antenna["radiation_resistance_ohm"], "radiation_resistance_ohm")
        reference_current = self.feed_current_a(self.current_moment_reference_frequency_hz,
                                                self.reference_power_kw)
        override = antenna.get("effective_length_m")
        self.effective_length_m = (float(override) if override is not None
                                   else self.current_moment_ref_a_m / reference_current)
        if self.effective_length_m <= 0:
            raise ValueError("antenna.effective_length_m must be positive")

    @staticmethod
    def _read_table(rows, name):
        if len(rows) < 2:
            raise ValueError(f"antenna.{name} needs at least two knots")
        data = np.asarray(rows, dtype=float)
        if data.shape[1] != 2 or np.any(~np.isfinite(data)):
            raise ValueError(f"antenna.{name} must be finite [frequency_hz, ohm] pairs")
        order = np.argsort(data[:, 0])
        data = data[order]
        if np.any(data[:, 0] <= 0) or np.any(data[:, 1] <= 0) or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError(f"antenna.{name} frequencies/values must be strictly increasing and positive")
        return data

    def _interpolate(self, table, frequency_hz):
        frequency = np.asarray(frequency_hz, dtype=float)
        if np.any(frequency <= 0):
            raise ValueError("Frequency must be positive")
        log_f = np.log10(frequency)
        return 10 ** np.interp(log_f, np.log10(table[:, 0]), np.log10(table[:, 1]))

    def input_resistance_ohm(self, frequency_hz):
        return self._interpolate(self.input_resistance, frequency_hz)

    def radiation_resistance_ohm(self, frequency_hz):
        return self._interpolate(self.radiation_resistance, frequency_hz)

    def feed_current_a(self, frequency_hz, power_kw=None):
        power = self.reference_power_kw if power_kw is None else float(power_kw)
        if power <= 0:
            raise ValueError("Transmit power must be positive")
        return np.sqrt(power * 1000.0 / self.input_resistance_ohm(frequency_hz))

    def current_moment_a_m(self, frequency_hz, power_kw=None):
        """Current moment at a frequency and active input power (A*m)."""
        return self.effective_length_m * self.feed_current_a(frequency_hz, power_kw)

    def radiation_efficiency(self, frequency_hz):
        return self.radiation_resistance_ohm(frequency_hz) / self.input_resistance_ohm(frequency_hz)

    def radiated_power_w(self, frequency_hz, power_kw=None):
        power = self.reference_power_kw if power_kw is None else float(power_kw)
        return power * 1000.0 * self.radiation_efficiency(frequency_hz)

    def beta_np_per_m(self, frequency_hz):
        omega = 2 * np.pi * np.asarray(frequency_hz, dtype=float)
        h_m = self.h_km * 1000.0
        return (1.0 / (2.0 * math.sqrt(2.0) * h_m)
                * (np.sqrt(omega * EPS0 / self.path_conductivity)
                   + np.sqrt(omega * EPS0 / self.ionosphere_conductivity)))

    def field_uv_m(self, frequency_hz, distance_m, power_kw=None, unit_convention=None):
        """Far-field vertical electric field in uV/m from equation (2.1.80)."""
        frequency = np.asarray(frequency_hz, dtype=float)
        distance = np.asarray(distance_m, dtype=float)
        if np.any(frequency <= 0) or np.any(distance <= 0):
            raise ValueError("Frequency and distance must be positive")
        unit = unit_convention or self.unit_convention
        if unit not in ("si", "engineering"):
            raise ValueError("unit_convention must be 'si' or 'engineering'")
        current_moment = self.current_moment_a_m(frequency, power_kw)
        beta = self.beta_np_per_m(frequency)
        if unit == "si":
            a_e = self.earth_radius_km * 1000.0
            h = self.h_km * 1000.0
            wavelength = C_LIGHT / frequency
            angle = distance / a_e
            denom = h * wavelength * np.sqrt(self.site_conductivity * a_e * np.sin(angle))
            field_v_m = math.sqrt(60.0) * math.pi * current_moment / denom * self.pattern_factor * np.exp(-beta * distance)
            return field_v_m * 1e6
        a_e = self.earth_radius_km
        h = self.h_km
        wavelength_km = C_LIGHT / frequency / 1000.0
        distance_km = distance / 1000.0
        angle = distance_km / a_e
        denom = h * wavelength_km * np.sqrt(self.site_conductivity * a_e * np.sin(angle))
        raw_engineering = math.sqrt(60.0) * math.pi * current_moment / denom * self.pattern_factor * np.exp(-beta * distance)
        # The formula evaluated with km is 1000*1000*sqrt(1000) times the
        # metre-valued field (h, lambda and a_e each lose one factor of 1000,
        # and the square root adds sqrt(1000)).  Convert that raw numeric to
        # the same uV/m scale as the SI branch instead of returning a value
        # 30 dB too high.
        return raw_engineering / math.sqrt(1000.0)

    def field_rms_pt(self, frequency_hz, distance_m, power_kw=None, unit_convention=None):
        """Air magnetic field B = E/c in pT for the antenna source model."""
        return self.field_uv_m(frequency_hz, distance_m, power_kw, unit_convention) * 1e-6 / C_LIGHT * 1e12

    def reference_field_uv_m(self, unit_convention=None):
        return self.field_uv_m(self.reference_frequency_hz, self.reference_distance_m,
                               self.reference_power_kw, unit_convention)

    def reference_field_rms_pt(self, unit_convention=None):
        return self.field_rms_pt(self.reference_frequency_hz, self.reference_distance_m,
                                 self.reference_power_kw, unit_convention)

    def fingerprint(self):
        payload = {
            "current_moment_a_m": self.current_moment_ref_a_m,
            "current_moment_reference_frequency_hz": self.current_moment_reference_frequency_hz,
            "effective_length_m": self.effective_length_m,
            "input_resistance_ohm": self.input_resistance.tolist(),
            "radiation_resistance_ohm": self.radiation_resistance.tolist(),
            "unit_convention": self.unit_convention,
            "pattern_factor": self.pattern_factor,
            "site_conductivity_s_m": self.site_conductivity,
            "path_conductivity_s_m": self.path_conductivity,
            "ionosphere_conductivity_s_m": self.ionosphere_conductivity,
            "earth_radius_km": self.earth_radius_km,
            "h_km": self.h_km,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

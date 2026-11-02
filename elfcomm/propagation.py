from __future__ import annotations

import hashlib
import json

import numpy as np

C_LIGHT = 299792458.0
EPS0 = 8.8541878128e-12


class PropagationModel:
    """Earth-ionosphere waveguide attenuation from Lu's textbook (3.2.25/3.2.33/3.2.36).

    The model computes the field attenuation rate beta(f) in dB/Mm from the
    ionospheric conductivity profile and applies it together with spherical
    1/d spreading:

        field_gain_db(f, d) = -20*log10(d/d_ref) - beta(f)*(d-d_ref)/1e6

    with d and d_ref in metres and beta in dB/Mm.  d_ref is the reference point
    (1950 km); at d_ref the gain is zero, so the absolute anchor stays at the
    reference field.  The formula is used inside the far-field training range
    (>= propagation_min_km); the LWPC table remains available as a cross-check.
    """

    def __init__(self, link):
        self.mode = link["ionosphere_mode"]
        if self.mode not in ("day", "night"):
            raise ValueError("ionosphere_mode must be 'day' or 'night'")
        self.scale = float(link["beta_scale"])
        if self.scale <= 0:
            raise ValueError("beta_scale must be positive")
        self.params = dict(link["ionosphere"])
        self.min_km = float(link["propagation_min_km"])
        self.reference_km = float(link["reference_environment"]["horizontal_distance_m"]) / 1000.0
        if self.reference_km <= 0 or self.min_km <= 0:
            raise ValueError("Propagation distances must be positive")
        self._validate_params()

    def _validate_params(self):
        if self.mode == "day":
            for key in ("H_km", "sigma0_km"):
                if self.params[key] <= 0:
                    raise ValueError(f"Invalid daytime ionosphere parameter: {key}")
        else:
            for key in ("H_km", "sigma0_km", "hE_km", "sigmaE_s_m"):
                if self.params[key] <= 0:
                    raise ValueError(f"Invalid nighttime ionosphere parameter: {key}")

    def beta_db_per_mm(self, frequency_hz):
        frequency = np.asarray(frequency_hz, dtype=float)
        if np.any(frequency <= 0):
            raise ValueError("Frequency must be positive")
        omega = 2 * np.pi * frequency
        k0 = omega / C_LIGHT
        if self.mode == "day":
            H = self.params["H_km"]
            sigma0 = self.params["sigma0_km"]
            sigma1 = self.params.get("sigma1_km", sigma0)
            h0 = H - sigma0 * np.log(2.5e5 / omega)
            h1 = h0 + 2 * sigma0 * np.log(2.39e4 / (frequency * sigma0))
            beta = 8.6858e6 * k0 * (np.pi / 4) * np.sqrt(h1 / h0) * (sigma0 / h0 + sigma1 / h1)
        else:
            H = self.params["H_km"]
            sigma0 = self.params["sigma0_km"]
            hE = self.params["hE_km"]
            sigmaE = self.params["sigmaE_s_m"]
            h0 = H - sigma0 * np.log(2.5e5 / omega)
            nE = np.sqrt(sigmaE / (omega * EPS0))
            beta = (0.1429 * frequency * np.sqrt(hE / h0)
                    * (sigma0 / h0 + 1.0 / (np.pi * k0 * nE * hE * 1e3)))
        return beta * self.scale

    def gain_db(self, frequency_hz, distance_m):
        distance_m = np.asarray(distance_m, dtype=float)
        distance_km = distance_m / 1000.0
        if np.any(~np.isfinite(distance_km)) or np.any(distance_km < self.min_km):
            raise ValueError(f"Propagation formula is used only for d >= {self.min_km:g} km")
        beta = self.beta_db_per_mm(frequency_hz)
        # beta is dB/Mm: convert the metre distance difference to Mm, not km.
        return -20.0 * np.log10(distance_km / self.reference_km) - beta * (distance_m - self.reference_km * 1000.0) / 1e6

    def power_ratio(self, frequency_hz, distance_m):
        return 10 ** (self.gain_db(frequency_hz, distance_m) / 10.0)

    def field_ratio(self, frequency_hz, distance_m):
        return 10 ** (self.gain_db(frequency_hz, distance_m) / 20.0)

    def fingerprint(self):
        payload = {"mode": self.mode, "scale": self.scale, "params": self.params,
                   "min_km": self.min_km, "reference_km": self.reference_km}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

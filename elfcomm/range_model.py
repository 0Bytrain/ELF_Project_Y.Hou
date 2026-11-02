from __future__ import annotations

import csv
import hashlib

import numpy as np

from .config import resolve_path


class RangeModel:
    """LWPC 30 Hz range curve, normalised at the configured reference point.

    The table stores the field amplitude A(d) in dB. ``gain_db`` returns
    ``A(d) - A(d_ref)`` (a field dB); power ratios must use ``/10`` and field
    ratios ``/20``. Only the training segment declared by the config is
    monotonised (the raw curve has a 0.06 dB pixel bump at 2600 km); the rest
    of the curve is kept for cross-checks.
    """

    def __init__(self, link):
        self.link = link
        self.path = resolve_path(link["range_table_path"])
        if not self.path.exists():
            raise FileNotFoundError(f"LWPC range table not found: {self.path}")
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("LWPC range table is empty")
        distance = np.array([float(row["range_km"]) for row in rows], dtype=float)
        amplitude = np.array([float(row["amplitude_db"]) for row in rows], dtype=float)
        order = np.argsort(distance)
        distance, amplitude = distance[order], amplitude[order]
        if len(distance) < 2 or np.any(np.diff(distance) <= 0):
            raise ValueError("LWPC range table must have strictly increasing range_km")
        if not np.isfinite(amplitude).all():
            raise ValueError("LWPC range table contains non-finite amplitudes")

        valid = [float(x) for x in link["range_table_valid_km"]]
        if len(valid) != 2 or not valid[0] < valid[1]:
            raise ValueError("range_table_valid_km must be [low, high]")
        if valid[0] < distance[0] or valid[1] > distance[-1]:
            raise ValueError("range_table_valid_km is not covered by the table")
        self.valid_km = (valid[0], valid[1])

        monotone = amplitude.copy()
        for i in range(1, len(monotone)):
            if valid[0] <= distance[i] <= valid[1]:
                monotone[i] = min(monotone[i], monotone[i - 1])
        self.distance_km = distance
        self.amplitude_db = monotone

        self.reference_km = float(link["reference_environment"]["horizontal_distance_m"]) / 1000.0
        if not distance[0] <= self.reference_km <= distance[-1]:
            raise ValueError("Reference distance is outside the LWPC range table")
        self.reference_db = float(np.interp(self.reference_km, distance, monotone))

    def gain_db(self, distance_m):
        """Field gain A(d) - A(d_ref) in dB; positive means stronger than reference."""
        d_km = np.asarray(distance_m, dtype=float) / 1000.0
        if np.any(~np.isfinite(d_km)) or np.any(d_km < self.distance_km[0]) or np.any(d_km > self.distance_km[-1]):
            raise ValueError(f"Distance outside LWPC range table [{self.distance_km[0]:g}, {self.distance_km[-1]:g}] km")
        return np.interp(d_km, self.distance_km, self.amplitude_db) - self.reference_db

    def power_ratio(self, distance_m):
        return 10 ** (self.gain_db(distance_m) / 10.0)

    def field_ratio(self, distance_m):
        return 10 ** (self.gain_db(distance_m) / 20.0)

    def fingerprint(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

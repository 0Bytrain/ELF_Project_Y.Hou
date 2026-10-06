from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import fingerprint, resolve_path, save_json

TABLE_SCHEMA = 2


def table_specification(cfg):
    source_hashes = {name: fingerprint((Path(__file__).parent / name).read_text(encoding="utf-8"))
                     for name in ("polar.py", "modem.py", "ber.py")}
    return {
        "schema": TABLE_SCHEMA, "polar": cfg["polar"],
        "modulations": cfg["actions"]["modulation"], "rates": cfg["actions"]["code_rate"],
        "simulation": {k: v for k, v in cfg["ber_table"].items() if k != "path"},
        "seed": cfg["seed"], "source_hashes": source_hashes,
        "receiver": "ideal_coherent_flat_equivalent_awgn",
    }


def row_key(modulation, rate, esn0_db):
    return f"{modulation}|{rate:.12g}|{esn0_db:.12g}"


def build_table(cfg, path=None, progress=print):
    from .ber import simulate_link

    path = resolve_path(path or cfg["ber_table"]["path"])
    spec = table_specification(cfg)
    signature = fingerprint(spec)
    data = {"signature": signature, "specification": spec, "rows": {}, "complete": False}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("signature") != signature:
            raise ValueError(f"BER table configuration/source mismatch: {path}. Use a new --table path.")
    jobs = [(m, r, s) for m in cfg["actions"]["modulation"] for r in cfg["actions"]["code_rate"]
            for s in cfg["ber_table"]["esn0_db"]]
    frames = cfg["ber_table"]
    for index, (modulation, rate, esn0_db) in enumerate(jobs):
        key = row_key(modulation, rate, esn0_db)
        if key in data["rows"]:
            continue
        result = simulate_link(modulation, rate, esn0_db, **cfg["polar"], seed=cfg["seed"] + 1009 * index,
                               min_frames=frames["min_frames"], max_frames=frames["max_frames"],
                               target_bit_errors=frames["target_bit_errors"])
        result.update(modulation=modulation, requested_code_rate=rate, esn0_db=esn0_db)
        data["rows"][key] = result
        save_json(path, data)
        if progress:
            progress(f"BER table {index + 1}/{len(jobs)} {modulation} R={rate:.3f} Es/N0={esn0_db:g} dB: "
                     f"{result['bit_errors']}/{result['total_bits']} bit errors", flush=True)
    data["complete"] = True
    save_json(path, data)
    return PerformanceTable(cfg, path)


def monotone_nonincreasing(values, weights):
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i + 1, float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            a, b = blocks[-2:]
            weight_sum = a[3] + b[3]
            blocks[-2:] = [[a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / weight_sum, weight_sum]]
    result = np.empty(len(values))
    for begin, end, value, _ in blocks:
        result[begin:end] = value
    return result


class PerformanceTable:
    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = resolve_path(path or cfg["ber_table"]["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"Build BER table first: python run.py build-table --config configs/default.json ({self.path})")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("signature") != fingerprint(table_specification(cfg)):
            raise ValueError("BER table mismatch: codec/modem/config changed; build a separate table")
        if not self.data.get("complete"):
            raise ValueError("Incomplete BER table: run build-table again to resume")
        self.knots = np.asarray(cfg["ber_table"]["esn0_db"], dtype=float)
        self.curves = {}
        for modulation in cfg["actions"]["modulation"]:
            for rate in cfg["actions"]["code_rate"]:
                rows = [self.data["rows"][row_key(modulation, rate, s)] for s in self.knots]
                bits = np.array([row["total_bits"] for row in rows])
                frames = np.array([row["total_frames"] for row in rows])
                ber = (np.array([row["bit_errors"] for row in rows]) + 0.5) / (bits + 1)
                fer = (np.array([row["frame_errors"] for row in rows]) + 0.5) / (frames + 1)
                self.curves[(modulation, rate)] = (
                    monotone_nonincreasing(ber, bits), monotone_nonincreasing(fer, frames))

    def lookup(self, modulation, rate, esn0_db):
        if not np.isfinite(esn0_db):
            raise ValueError("Es/N0 must be finite")
        if esn0_db < self.knots[0]:
            return {"ber_estimate": 0.5, "fer_estimate": 1.0, "table_status": "below_table_conservative_fallback",
                    "reliability_certified": False}
        ber, fer = self.curves[(modulation, rate)]
        return {
            "ber_estimate": float(10 ** np.interp(esn0_db, self.knots, np.log10(ber))),
            "fer_estimate": float(10 ** np.interp(esn0_db, self.knots, np.log10(fer))),
            "table_status": "interpolated_estimate" if esn0_db <= self.knots[-1] else "above_table_endpoint_estimate",
            "reliability_certified": False,
        }


def evaluate_action(action, channel_result, table, cfg):
    n = cfg["polar"]["n"]
    k = round(n * action.code_rate)
    payload = k - cfg["polar"]["crc_bits"]
    modulation_symbols = n / action.bits_per_symbol
    frame_seconds = (modulation_symbols + cfg["objective"]["preamble_symbols"]) / action.symbol_rate
    estimate = table.lookup(action.modulation, action.code_rate, channel_result["esn0_db"])
    raw_information_rate = action.symbol_rate * action.bits_per_symbol * payload / n
    goodput = payload * (1 - estimate["fer_estimate"]) / frame_seconds
    return {
        **action.as_dict(), **channel_result, **estimate, "actual_code_rate": k / n,
        "payload_code_rate": payload / n, "information_bits_with_crc": k, "payload_bits": payload,
        "nominal_information_rate_bps": raw_information_rate, "frame_duration_s": frame_seconds,
        "goodput_bps_estimate": float(goodput),
        "meets_ber_target_estimate": bool(estimate["ber_estimate"] <= cfg["objective"]["target_ber"]),
        "meets_rate_target_estimate": bool(goodput >= cfg["objective"]["target_goodput_bps"]),
    }


def utility(metrics, cfg, switched=False):
    objective = cfg["objective"]
    if not metrics["legal"]:
        return float(objective["outage_reward"])
    goodput_ratio = metrics["goodput_bps_estimate"] / objective["target_goodput_bps"]
    violation = max(0.0, np.log10(metrics["ber_estimate"] / objective["target_ber"]))
    return float(objective["reward_goodput_weight"] * goodput_ratio
                 - objective["reward_ber_violation_weight"] * min(violation, 5.0)
                 - objective["reward_rate_shortfall_weight"] * max(0.0, 1 - goodput_ratio)
                 - objective["switch_penalty"] * switched)

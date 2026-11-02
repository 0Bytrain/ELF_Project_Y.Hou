from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np

from .codecs import build_codec
from .config import coding_modes, fingerprint, resolve_path, save_json, waveform_mode_index

TABLE_SCHEMA = 3
PARITY_COMPATIBILITY = ('c24cca17e09f336a2e606c03078755e0eaeadb45c0828fc5ea30fc8c3fc0e35e', '18eeb8706bd1fb094239f8f420ae056b385944458a426b584c05370f2f4c10b1')


def table_specification(cfg):
    source_hashes = {name: fingerprint((Path(__file__).parent / name).read_text(encoding="utf-8"))
                     for name in ("polar.py", "codecs.py", "modem.py", "ber.py")}
    return {
        "schema": TABLE_SCHEMA, "polar": cfg["polar"],
        "coding_modes": [list(mode) for mode in coding_modes(cfg)],
        "modulations": cfg["actions"]["modulation"],
        "simulation": {k: v for k, v in cfg["ber_table"].items() if k not in ("path", "coarse_table_path", "existing_only")},
        "seed": cfg["seed"], "source_hashes": source_hashes,
        "receiver": "ideal_coherent_flat_equivalent_awgn",
    }


def compatible_table_signatures(cfg):
    spec = table_specification(cfg)
    signatures = {fingerprint(spec)}
    if spec["source_hashes"]["codecs.py"] == PARITY_COMPATIBILITY[1]:
        legacy = copy.deepcopy(spec)
        legacy["source_hashes"]["codecs.py"] = PARITY_COMPATIBILITY[0]
        signatures.add(fingerprint(legacy))
    return signatures


def row_key(coding_scheme, modulation, rate, esn0_db):
    return f"{coding_scheme}|{modulation}|{rate:.12g}|{esn0_db:.12g}"


def refinement_jobs(cfg, coarse_rows):
    options = cfg["ber_table"].get("refinement", {})
    if not options.get("enabled", False):
        return []
    knots = np.asarray(cfg["ber_table"]["esn0_db"], dtype=float)
    jobs = []
    for scheme, rate in coding_modes(cfg):
        for modulation in cfg["actions"]["modulation"]:
            rows = [coarse_rows[row_key(scheme, modulation, rate, snr)] for snr in knots]
            bits = np.array([r["total_bits"] for r in rows])
            ber = monotone_nonincreasing(np.array([r["bit_errors"] for r in rows]) / bits, bits)
            crossings = np.flatnonzero(ber <= options["target_ber"])
            if crossings.size:
                upper = int(crossings[0])
            elif ber[-1] <= options["near_target_multiplier"] * options["target_ber"]:
                upper = len(knots) - 1
            else:
                continue
            if upper == 0:
                points = [knots[0]]
            else:
                points = np.linspace(knots[upper - 1], knots[upper], options["subdivisions"] + 1)[1:]
            jobs.extend((scheme, modulation, rate, float(snr)) for snr in points)
    return jobs


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
    elif cfg["ber_table"].get("coarse_table_path"):
        coarse_path = resolve_path(cfg["ber_table"]["coarse_table_path"])
        if coarse_path == path:
            raise ValueError("Refined and coarse BER tables must use different paths")
        if coarse_path.exists():
            coarse_cfg = copy.deepcopy(cfg)
            coarse_cfg["ber_table"].pop("refinement", None)
            coarse_cfg["ber_table"].pop("coarse_table_path", None)
            expected = table_specification(coarse_cfg)
            signatures = {fingerprint(expected)}
            if expected["source_hashes"]["codecs.py"] == PARITY_COMPATIBILITY[1]:
                legacy = copy.deepcopy(expected)
                legacy["source_hashes"]["codecs.py"] = PARITY_COMPATIBILITY[0]
                signatures.add(fingerprint(legacy))
            coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
            if coarse.get("signature") not in signatures:
                raise ValueError("Coarse BER table mismatch; use a compatible source table")
            data["rows"] = copy.deepcopy(coarse["rows"])
            data["coarse_source_signature"] = coarse["signature"]
            if progress:
                progress(f"Reusing {len(data['rows'])} coarse BER points; original table is preserved.", flush=True)
    jobs = [(scheme, modulation, rate, snr)
            for scheme, rate in coding_modes(cfg)
            for modulation in cfg["actions"]["modulation"]
            for snr in cfg["ber_table"]["esn0_db"]]
    frames = cfg["ber_table"]
    for index, (scheme, modulation, rate, snr) in enumerate(jobs):
        key = row_key(scheme, modulation, rate, snr)
        if key in data["rows"]:
            continue
        result = simulate_link(scheme, modulation, rate, snr, **cfg["polar"],
                               seed=cfg["seed"] + 1009 * index,
                               min_frames=frames["min_frames"], max_frames=frames["max_frames"],
                               target_bit_errors=frames["target_bit_errors"])
        result.update(coding_scheme=scheme, modulation=modulation, requested_code_rate=rate, esn0_db=snr)
        data["rows"][key] = result
        save_json(path, data)
        if progress:
            progress(f"Coarse BER table {index + 1}/{len(jobs)}: {result['bit_errors']}/{result['total_bits']} errors", flush=True)
    options = frames.get("refinement", {})
    if options.get("enabled", False):
        if "coarse_rows" not in data:
            data["coarse_rows"] = copy.deepcopy(data["rows"])
            data["refinement_jobs"] = refinement_jobs(cfg, data["coarse_rows"])
            data["refinement_completed"] = []
        save_json(path, data)
        fine_jobs = data["refinement_jobs"]
        completed = set(data["refinement_completed"])
        if progress:
            progress(f"Refining {len(fine_jobs)} transition points, up to {options['max_frames']} frames per point; progress is resumable.", flush=True)
        for index, (scheme, modulation, rate, snr) in enumerate(fine_jobs):
            key = row_key(scheme, modulation, rate, snr)
            if key in completed:
                continue
            if progress:
                progress(f"Starting refinement {index + 1}/{len(fine_jobs)} {scheme}/{modulation} "
                         f"R={rate:.3f} Es/N0={snr:g} dB (up to {options['max_frames']} frames)", flush=True)
            result = simulate_link(scheme, modulation, rate, snr, **cfg["polar"],
                                   seed=cfg["seed"] + 1000003 + 1009 * index,
                                   min_frames=options["min_frames"], max_frames=options["max_frames"],
                                   target_bit_errors=options["target_bit_errors"])
            result.update(coding_scheme=scheme, modulation=modulation, requested_code_rate=rate,
                          esn0_db=snr, refinement=True)
            data["rows"][key] = result
            data["refinement_completed"].append(key)
            completed.add(key)
            save_json(path, data)
            if progress:
                progress(f"Refined BER {index + 1}/{len(fine_jobs)} {scheme}/{modulation} R={rate:.3f} "
                         f"Es/N0={snr:g}: {result['bit_errors']}/{result['total_bits']} errors", flush=True)
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
        if self.data.get("signature") not in compatible_table_signatures(cfg):
            raise ValueError("BER table mismatch: codec/modem/config changed; use a compatible table")
        if not self.data.get("complete"):
            raise ValueError("Incomplete BER table: run build-table again to resume")
        self.knots = np.asarray(cfg["ber_table"]["esn0_db"], dtype=float)
        self.curves, self.bit_budgets, self.curve_knots = {}, {}, {}
        grouped = {}
        for encoded, row in self.data["rows"].items():
            scheme, modulation, rate, snr = encoded.split("|")
            grouped.setdefault((scheme, modulation, float(rate)), []).append((float(snr), row))
        for scheme, rate in coding_modes(cfg):
            for modulation in cfg["actions"]["modulation"]:
                key = (scheme, modulation, rate)
                matches = [items for k, items in grouped.items()
                           if k[:2] == key[:2] and abs(k[2] - rate) < 1e-10]
                if len(matches) != 1:
                    raise ValueError(f"BER table is missing/ambiguous for {key}")
                ordered = sorted(matches[0], key=lambda item: item[0])
                knots = np.array([snr for snr, _ in ordered])
                if any(not np.any(np.isclose(knots, snr, atol=1e-10, rtol=0)) for snr in self.knots):
                    raise ValueError(f"BER table is missing a configured coarse knot for {key}")
                rows = [row for _, row in ordered]
                bits = np.array([row["total_bits"] for row in rows])
                frames = np.array([row["total_frames"] for row in rows])
                ber = np.array([row["bit_errors"] for row in rows]) / bits
                fer = np.array([row["frame_errors"] for row in rows]) / frames
                self.curves[key] = (monotone_nonincreasing(ber, bits), monotone_nonincreasing(fer, frames))
                self.bit_budgets[key], self.curve_knots[key] = bits, knots

    def lookup_ber_many(self, coding_scheme, modulation, rate, esn0_db):
        values = np.asarray(esn0_db, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Es/N0 must be finite")
        ber, _ = self.curves[(coding_scheme, modulation, rate)]
        return np.interp(values, self.curve_knots[(coding_scheme, modulation, rate)], ber, left=0.5)

    def lookup(self, coding_scheme, modulation, rate, esn0_db):
        if not np.isfinite(esn0_db):
            raise ValueError("Es/N0 must be finite")
        key = (coding_scheme, modulation, rate)
        knots = self.curve_knots[key]
        if esn0_db < knots[0]:
            return {"ber_estimate": 0.5, "fer_estimate": 1.0, "table_status": "below_table_conservative_fallback",
                    "reliability_certified": False}
        key = (coding_scheme, modulation, rate)
        ber, fer = self.curves[key]
        estimate = float(np.interp(esn0_db, knots, ber))
        upper = int(np.clip(np.searchsorted(knots, esn0_db), 0, len(knots) - 1))
        lower = upper if knots[upper] == esn0_db or upper == 0 or esn0_db >= knots[-1] else upper - 1
        # Zero errors are an unresolved observation, not evidence of exact BER=0.
        # Linear interpolation permits zero counts without introducing a fictitious floor.
        uses_zero_counts = bool(ber[lower] == 0 or ber[upper] == 0)
        resolution = float(1.0 / min(self.bit_budgets[key][lower], self.bit_budgets[key][upper]))
        limited = bool(estimate <= resolution)
        return {
            "ber_estimate": estimate,
            "fer_estimate": float(np.interp(esn0_db, knots, fer)),
            "ber_resolution_limited": limited,
            "ber_resolution_limit": resolution,
            "ber_interpolation_uses_zero_counts": uses_zero_counts,
            "ber_estimator": "empirical_monotone_linear_v2",
            "table_status": "interpolated_estimate" if esn0_db <= knots[-1] else "above_table_endpoint_estimate",
            "reliability_certified": False,
        }


def load_or_build_table(cfg, path=None, progress=print):
    """Load a compatible BER table; create/resume it only when it is missing/incomplete.

    A signature mismatch is not silently overwritten: the old table belongs to
    another codec/actionset and must be rebuilt under a new path.
    """
    target = resolve_path(path or cfg["ber_table"]["path"])
    if cfg["ber_table"].get("existing_only", False):
        table = PerformanceTable(cfg, target)
        if progress:
            progress(f"Using existing BER table: {target} ({len(table.data['rows'])} points); "
                     "skipping table building and refinement.", flush=True)
        return table
    try:
        return PerformanceTable(cfg, target)
    except FileNotFoundError:
        progress(f"BER table not found: {target}; building it now (this can take a while).", flush=True)
        return build_table(cfg, target, progress=progress)
    except ValueError as error:
        if "Incomplete BER table" not in str(error):
            raise
        progress(f"BER table is incomplete: {target}; resuming the build.", flush=True)
        return build_table(cfg, target, progress=progress)


def evaluate_action(action, channel_result, table, cfg):
    codec = build_codec(action.coding_scheme, n=cfg["polar"]["n"], rate=action.code_rate,
                        crc_bits=cfg["polar"]["crc_bits"], list_size=cfg["polar"]["list_size"])
    payload = codec.payload_bits
    coded_bits = codec.coded_bits
    modulation_symbols = coded_bits / action.bits_per_symbol
    frame_seconds = (modulation_symbols + cfg["objective"]["preamble_symbols"]) / action.symbol_rate
    estimate = table.lookup(action.coding_scheme, action.modulation,
                            action.code_rate, channel_result["esn0_db"])
    raw_information_rate = action.symbol_rate * action.bits_per_symbol * payload / coded_bits
    goodput = payload * (1 - estimate["fer_estimate"]) / frame_seconds
    return {
        **action.as_dict(), **channel_result, **estimate,
        "coding_scheme": action.coding_scheme, "coding_mode": list(action.coding_mode),
        "waveform_mode": list(action.waveform_mode), "waveform_mode_index": waveform_mode_index(action, cfg),
        "actual_code_rate": codec.actual_rate,
        "payload_code_rate": codec.payload_rate, "information_bits_with_crc": codec.information_bits,
        "payload_bits": payload, "coded_bits": coded_bits,
        "nominal_information_rate_bps": raw_information_rate, "frame_duration_s": frame_seconds,
        "goodput_bps_estimate": float(goodput),
        "meets_ber_target_estimate": bool(estimate["ber_estimate"] <= cfg["objective"]["target_ber"]),
        "meets_rate_target_estimate": bool(goodput >= cfg["objective"]["target_goodput_bps"]),
    }


def utility(metrics, cfg, switched=False, previous_power=None, baseline_ber=None):
    objective = cfg["objective"]
    if not metrics["legal"]:
        return float(objective["outage_reward"])
    goodput_ratio = metrics["goodput_bps_estimate"] / objective["target_goodput_bps"]
    powers = cfg["actions"]["tx_power_kw"]
    power_norm = (metrics["tx_power_kw"] - min(powers)) / max(max(powers) - min(powers), 1e-12)
    ber_limit = objective["target_ber"]
    if objective.get("mode") == "ber_constrained_min_power":
        if metrics["ber_estimate"] > ber_limit:
            violation = np.log10(metrics["ber_estimate"] / ber_limit)
            return float(-1.0 - objective["reward_ber_violation_weight"] * violation)
        # Every feasible action outranks every infeasible one. Among feasible
        # actions power dominates bounded throughput/switch tie breakers.
        normalized = sorted(set((float(p) - min(powers)) / max(max(powers) - min(powers), 1e-12)
                                for p in powers))
        gap = min(np.diff(normalized)) if len(normalized) > 1 else 1.0
        tie = 0.2 * gap
        return float(1.0 - power_norm + tie * np.clip(goodput_ratio, 0.0, 1.0)
                     - min(objective["switch_penalty"], tie) * bool(switched))
    violation = max(0.0, np.log10(max(metrics["ber_estimate"], 1e-300) / ber_limit))
    return float(objective["reward_goodput_weight"] * goodput_ratio
                 - objective["reward_ber_violation_weight"] * min(violation, 5.0)
                 - objective["reward_rate_shortfall_weight"] * max(0.0, 1 - goodput_ratio)
                 - objective["switch_penalty"] * switched
                 - objective.get("reward_power_weight", 0.0) * power_norm)

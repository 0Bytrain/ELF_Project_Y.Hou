"""Monte Carlo payload BER for the selected coding scheme.

The frame contract is shared by Polar and CC: ``payload_bits`` excludes the
common CRC-11, and ``fer`` counts payload errors or CRC rejection.  The codec
reports its own actual rate and coded length, so nominal rates are not silently
treated as exact rates.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import beta
from .codecs import build_codec
from .modem import transmit_llr, bits_per_symbol, MODEM_VERSION

BER_SCHEMA_VERSION = "payload-counts-multicodec-v1"


def binomial_upper95(errors, trials):
    """One-sided exact Clopper-Pearson limit for fixed-budget iid trials."""
    if trials <= 0 or errors < 0 or errors > trials:
        raise ValueError("Require 0 <= errors <= positive trials")
    if errors == trials:
        return 1.0
    return float(beta.ppf(0.95, errors + 1, trials - errors))


def simulate_link(coding_scheme, modulation, rate, esn0_db, n=256, crc_bits=11,
                  list_size=8, seed=0, min_frames=32, max_frames=256,
                  target_bit_errors=100):
    """Count actual decoded payload errors, including failed-CRC frames.

    ``fer`` counts payload frame errors OR CRC rejection, as required for
    throughput. ``ber`` counts payload mismatches even on rejected frames.
    Bit upper limits assume independent bits and are merely nominal because
    errors within a code word are correlated. Frame limits can conservatively
    bound BER only with independent frames and a fixed sample budget. With
    adaptive target-error stopping both reported limits are descriptive and
    must not be used to claim verified reliability.
    """
    bits_per_symbol(modulation)
    if not (1 <= min_frames <= max_frames) or target_bit_errors < 1:
        raise ValueError("Require 1 <= min_frames <= max_frames and positive target errors")
    codec = build_codec(coding_scheme, n=n, rate=rate, crc_bits=crc_bits,
                        list_size=list_size)
    rng = np.random.default_rng(seed)
    bit_errors = total_bits = frame_errors = total_frames = crc_failures = 0
    payload_frame_errors = 0
    stop_reason = "max_frames"
    for _ in range(max_frames):
        payload = rng.integers(0, 2, size=codec.payload_bits, dtype=np.uint8)
        llr = transmit_llr(codec.encode(payload), modulation, esn0_db, rng)
        decoded, crc_ok = codec.decode_with_status(llr)
        errors = int(np.count_nonzero(decoded != payload))
        bit_errors += errors
        total_bits += len(payload)
        total_frames += 1
        payload_frame_errors += int(errors > 0)
        crc_failures += int(not crc_ok)
        frame_errors += int(errors > 0 or not crc_ok)
        if total_frames >= min_frames and bit_errors >= target_bit_errors:
            stop_reason = "target_bit_errors" if total_frames < max_frames else "max_frames"
            break
    fixed_budget = min_frames == max_frames
    return {
        "coding_scheme": coding_scheme, "nominal_rate": float(rate),
        "actual_rate": codec.actual_rate, "payload_rate": codec.payload_rate,
        "modulation": modulation.upper(), "esn0_db": float(esn0_db),
        "n": n, "coded_bits_per_frame": codec.coded_bits,
        "information_bits_per_frame": codec.information_bits,
        "payload_bits_per_frame": codec.payload_bits,
        "crc_bits": crc_bits, "list_size": list_size, "seed": int(seed),
        "bit_errors": bit_errors, "total_bits": total_bits,
        "frame_errors": frame_errors, "payload_frame_errors": payload_frame_errors,
        "total_frames": total_frames, "crc_failures": crc_failures,
        "ber": bit_errors / total_bits, "fer": frame_errors / total_frames,
        "ber_upper95": binomial_upper95(bit_errors, total_bits),
        "fer_upper95": binomial_upper95(frame_errors, total_frames),
        "zero_bit_errors_observed": bit_errors == 0,
        "stop_reason": stop_reason, "fixed_frame_budget": fixed_budget,
        "confidence_note": (
            "Bit CI is nominal independent-bit approximation; coded bits are correlated. "
            + ("Frame CI is fixed-budget iid-frame CP; it conservatively bounds payload BER."
               if fixed_budget else
               "Adaptive stopping: all CP endpoints are descriptive, not reliability certificates.")
        ),
        "modem_version": MODEM_VERSION, "schema_version": BER_SCHEMA_VERSION,
    }

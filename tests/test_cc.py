import numpy as np
import pytest

from elfcomm.codecs import (
    CC_RATES, ConvolutionalCode, PolarCodec, build_codec, crc11_append, crc11_check,
)
from elfcomm.config import load_config
from elfcomm.performance import row_key, table_specification


@pytest.mark.parametrize("rate", CC_RATES)
def test_cc_noiseless_roundtrip_at_configured_rates(rate):
    code = ConvolutionalCode(n=256, rate=rate)
    rng = np.random.default_rng(2026)
    for _ in range(3):
        payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
        encoded = code.encode(payload)
        assert encoded.size == code.coded_bits == 256
        decoded, passed = code.decode_with_status(60.0 * (1.0 - 2.0 * encoded))
        np.testing.assert_array_equal(decoded, payload)
        assert passed


def test_cc_zero_payload_produces_zero_codeword():
    code = ConvolutionalCode(n=256, rate=0.5)
    encoded = code.encode(np.zeros(code.payload_bits, dtype=np.uint8))
    np.testing.assert_array_equal(encoded, np.zeros(256, dtype=np.uint8))


def _reference_viterbi_terminated(code, llr):
    """Independent K=7 Viterbi with the known zero-tail final state."""
    metrics = np.full(64, -np.inf)
    metrics[0] = 0.0
    back_state = np.empty((code.k_encoder, 64), dtype=np.int16)
    back_bit = np.empty((code.k_encoder, 64), dtype=np.uint8)
    for t in range(code.k_encoder):
        new_metrics = np.full(64, -np.inf)
        for state in range(64):
            if not np.isfinite(metrics[state]):
                continue
            for bit in (0, 1):
                next_state = ((state >> 1) | (bit << 5)) & 63
                window = (bit << 6) | state
                expected0 = (window & 0o171).bit_count() & 1
                expected1 = (window & 0o133).bit_count() & 1
                branch = 0.5 * ((1 - 2 * expected0) * llr[2 * t]
                                + (1 - 2 * expected1) * llr[2 * t + 1])
                score = metrics[state] + branch
                if score > new_metrics[next_state]:
                    new_metrics[next_state] = score
                    back_state[t, next_state] = state
                    back_bit[t, next_state] = bit
        metrics = new_metrics
    state = 0
    bits = np.empty(code.k_encoder, dtype=np.uint8)
    for t in range(code.k_encoder - 1, -1, -1):
        bits[t] = back_bit[t, state]
        state = int(back_state[t, state])
    return bits


def test_cc_viterbi_forces_terminated_zero_state():
    code = ConvolutionalCode(n=256, rate=0.5)
    rng = np.random.default_rng(20261025)
    for _ in range(4):
        llr = rng.normal(0.0, 3.0, code.base_length)
        np.testing.assert_array_equal(code._decode_base(llr),
                                      _reference_viterbi_terminated(code, llr))


def test_cc_encoder_matches_standard_171_133_tap_order():
    """Regression for the left-shift state bug.

    The reference below is the standard K=7 shift register: the newest input
    is bit 6 of the generator window, and the previous six inputs live in bits
    5..0 with the newest previous input in bit 5.
    """
    code = ConvolutionalCode(n=256, rate=0.5)
    rng = np.random.default_rng(20261024)
    for _ in range(10):
        bits = rng.integers(0, 2, code.k_encoder, dtype=np.uint8)
        state = 0
        expected = np.empty(2 * code.k_encoder, dtype=np.uint8)
        for i, bit in enumerate(bits):
            window = (int(bit) << 6) | state
            expected[2 * i] = (window & 0o171).bit_count() & 1
            expected[2 * i + 1] = (window & 0o133).bit_count() & 1
            state = ((state >> 1) | (int(bit) << 5)) & 0x3F
        np.testing.assert_array_equal(code._encode_base(bits), expected)


def test_cc_payload_and_actual_rate_are_recorded_per_mode():
    expected = {0.5: (111, 0.4765625), 2 / 3: (154, 0.64453125), 0.75: (175, 0.7265625)}
    for rate, (payload_bits, actual_rate) in expected.items():
        code = ConvolutionalCode(n=256, rate=rate)
        assert code.payload_bits == payload_bits
        assert code.actual_rate == pytest.approx(actual_rate)
        assert code.coded_bits == 256
        assert code.payload_rate == pytest.approx(payload_bits / 256)


def test_cc_detects_a_corrupted_codeword_via_crc():
    code = ConvolutionalCode(n=256, rate=0.5)
    rng = np.random.default_rng(7)
    payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
    encoded = code.encode(payload)
    # Flip a large block so the Viterbi output is not the original CRC word.
    corrupted = encoded.copy()
    corrupted[:40] ^= 1
    decoded, passed = code.decode_with_status(2.0 * (1.0 - 2.0 * corrupted))
    assert decoded.size == code.payload_bits
    assert not passed or np.array_equal(decoded, payload)


def test_crc11_helper_matches_existing_polar_polynomial():
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    word = crc11_append(payload)
    assert word.size == payload.size + 11
    assert crc11_check(word)
    bad = word.copy()
    bad[0] ^= 1
    assert not crc11_check(bad)


def test_polar_codec_adapter_preserves_existing_payload_sizes():
    code = PolarCodec(n=256, rate=0.75)
    assert code.payload_bits == 181
    assert code.coded_bits == 256
    assert code.actual_rate == pytest.approx(0.75)
    assert build_codec("Polar", n=256, rate=0.75).payload_bits == 181
    with pytest.raises(ValueError):
        build_codec("LDPC", n=256, rate=0.5)


def test_table_key_and_signature_include_coding_scheme():
    cfg = load_config()
    assert row_key("Polar", "BPSK", 0.5, 3.0) != row_key("CC", "BPSK", 0.5, 3.0)
    spec = table_specification(cfg)
    assert ["Polar", 0.3333333333333333] in spec["coding_modes"]
    assert ["CC", 0.75] in spec["coding_modes"]
    assert "codecs.py" in spec["source_hashes"]


def test_missing_or_incomplete_table_is_built_but_mismatch_is_not_overwritten(monkeypatch):
    import elfcomm.performance as performance

    monkeypatch.setattr(performance, "build_table",
                        lambda cfg, path=None, progress=print: "built")
    monkeypatch.setattr(performance, "PerformanceTable",
                        lambda cfg, path=None: (_ for _ in ()).throw(FileNotFoundError("missing")))
    assert performance.load_or_build_table({}, "missing.json") == "built"
    monkeypatch.setattr(performance, "PerformanceTable",
                        lambda cfg, path=None: (_ for _ in ()).throw(ValueError("Incomplete BER table: missing")))
    assert performance.load_or_build_table({}, "missing.json") == "built"
    monkeypatch.setattr(performance, "PerformanceTable",
                        lambda cfg, path=None: (_ for _ in ()).throw(ValueError("BER table mismatch: other axis")))
    with pytest.raises(ValueError, match="mismatch"):
        performance.load_or_build_table({}, "missing.json")

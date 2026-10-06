import itertools

import numpy as np
import pytest

from elfcomm.polar import PolarCode, polar_transform


@pytest.mark.parametrize("rate", [1 / 3, 0.5, 2 / 3, 0.75])
def test_noiseless_roundtrip_at_requested_rates(rate):
    code = PolarCode(n=256, rate=rate, list_size=8)
    rng = np.random.default_rng(112)
    for _ in range(3):
        payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
        encoded = code.encode(payload)
        result, passed = code.decode_with_status(60.0 * (1.0 - 2.0 * encoded))
        np.testing.assert_array_equal(result, payload)
        assert passed
    assert code.actual_rate == code.k / 256
    assert code.payload_rate == (code.k - 11) / 256


def test_crc_detects_each_single_bit_corruption():
    code = PolarCode()
    rng = np.random.default_rng(5)
    payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
    information = code.add_crc(payload)
    assert code.check_crc(information)
    for j in range(code.k):
        corrupt = information.copy()
        corrupt[j] ^= 1
        assert not code.check_crc(corrupt)


def test_crc_matches_polynomial_long_division():
    code = PolarCode(n=32, rate=0.75)
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
    remainder = np.concatenate((payload, np.zeros(11, dtype=np.uint8)))
    generator = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
    for j in range(payload.size):
        if remainder[j]:
            remainder[j : j + 12] ^= generator
    np.testing.assert_array_equal(code.add_crc(payload)[-11:], remainder[-11:])


def test_full_list_small_code_matches_exhaustive_ml():
    code = PolarCode(n=8, rate=0.5, crc_bits=0, list_size=16)
    payloads = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    words = np.array([code.encode(p) for p in payloads])
    rng = np.random.default_rng(432)
    for _ in range(15):
        llr = rng.normal(size=8) * 3
        costs = np.logaddexp(0, -(1 - 2 * words.astype(float)) * llr).sum(axis=1)
        np.testing.assert_array_equal(code.decode(llr), payloads[np.argmin(costs)])


def test_seeded_awgn_good_channel_improves_over_bad_channel():
    code = PolarCode(n=64, rate=0.5, list_size=8)
    rng = np.random.default_rng(13)
    bad_errors = good_errors = 0
    for _ in range(16):
        payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
        symbols = 1.0 - 2.0 * code.encode(payload)
        noise = rng.standard_normal(code.n)
        for sigma, category in ((2.5, "bad"), (0.5, "good")):
            decoded = code.decode(2.0 * (symbols + sigma * noise) / sigma**2)
            errors = np.count_nonzero(payload != decoded)
            if category == "bad":
                bad_errors += errors
            else:
                good_errors += errors
    assert good_errors == 0
    assert bad_errors > good_errors


def test_transform_is_its_own_inverse():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, 256, dtype=np.uint8)
    np.testing.assert_array_equal(polar_transform(polar_transform(bits)), bits)


def test_invalid_parameters_and_llrs():
    with pytest.raises(ValueError):
        PolarCode(n=30)
    with pytest.raises(ValueError):
        PolarCode(n=16, rate=0.5)
    with pytest.raises(ValueError):
        PolarCode(crc_bits=8)
    with pytest.raises(ValueError):
        PolarCode().decode(np.full(256, np.nan))

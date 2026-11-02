import numpy as np
import pytest

from elfcomm.codecs import BCH_RATES, BCHCode, build_codec


@pytest.mark.parametrize("rate", BCH_RATES)
def test_bch_noiseless_roundtrip_and_extension(rate):
    code = BCHCode(rate=rate)
    rng = np.random.default_rng(2026)
    for _ in range(2):
        payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
        encoded = code.encode(payload)
        assert encoded.size == 256
        decoded, passed = code.decode_with_status(60.0 * (1.0 - 2.0 * encoded))
        np.testing.assert_array_equal(decoded, payload)
        assert passed
    assert code.coded_bits == 256
    assert code.actual_rate == pytest.approx(code.k_bch / 256)
    assert code.payload_rate == pytest.approx((code.k_bch - 11) / 256)


@pytest.mark.parametrize("rate", BCH_RATES)
def test_bch_hard_decoder_corrects_up_to_t_errors(rate):
    code = BCHCode(rate=rate)
    rng = np.random.default_rng(77)
    payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
    encoded = code.encode(payload)
    for error_count in (0, 1, 2, code.t):
        received = encoded.copy()
        positions = rng.choice(256, size=error_count, replace=False)
        received[positions] ^= 1
        decoded, passed = code.decode_with_status(60.0 * (1.0 - 2.0 * received))
        np.testing.assert_array_equal(decoded, payload)
        assert passed


def test_bch_extension_parity_is_used_for_t_plus_one_detection():
    code = BCHCode(rate=BCH_RATES[0])
    rng = np.random.default_rng(20261026)
    payload = rng.integers(0, 2, code.payload_bits, dtype=np.uint8)
    encoded = code.encode(payload)
    positions = rng.choice(255, size=code.t, replace=False)
    received = encoded.copy()
    received[positions] ^= 1
    # t base errors plus a corrupted extension bit is t+1 errors and must not
    # be reported as successfully corrected by the extended code.
    received[255] ^= 1
    _, passed = code.decode_with_status(60.0 * (1.0 - 2.0 * received))
    assert not passed


def test_bch_dimensions_are_explicit_not_invented():
    expected = {0.3568627450980392: (91, 21),
                0.5137254901960784: (131, 16),
                0.6705882352941176: (171, 11),
                0.7490196078431373: (191, 8)}
    for rate, (k, t) in expected.items():
        code = BCHCode(rate=rate)
        assert code.k_bch == k
        assert code.t == t
    with pytest.raises(ValueError):
        build_codec("BCH", n=64, rate=BCH_RATES[0])

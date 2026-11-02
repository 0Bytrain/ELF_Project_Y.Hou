"""Coding-scheme registry and the Phase-1 convolutional code.

The common contract is a 256-bit coded frame (or the configured ``n``):

* ``payload_bits`` information bits after the common CRC-11;
* ``information_bits`` information bits including CRC-11;
* ``coded_bits`` transmitted coded bits;
* ``actual_rate = information_bits / coded_bits`` (CRC included, tail excluded);
* ``payload_rate = payload_bits / coded_bits``.

Polar keeps its existing CRC-aided SCL implementation.  The convolutional
code is a K=7, rate-1/2, zero-terminated code with generators (171,133)_8 and a
deterministic puncturing mask for the nominal 2/3 and 3/4 modes.  It is a
reproducible research baseline, not a claim of CCSDS conformance or ELF field
deployment.  Puncturing is intentionally documented here because the exact
mask changes the result: the last 2*tail_bits positions are always retained and
the remaining positions are selected uniformly.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .polar import PolarCode

CRC_BITS = 11
CRC_POLYNOMIAL = 0x621
POLAR_RATES = (1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75)
CC_RATES = (0.5, 2.0 / 3.0, 0.75)
# Binary BCH(255,k), extended with one overall parity bit to 256.  k=91 is
# used for the near-1/3 mode because the requested (255,87) dimension is not
# produced by the primitive polynomial x^8+x^4+x^3+x^2+1 for consecutive roots.
BCH_RATES = (91.0 / 255.0, 131.0 / 255.0, 171.0 / 255.0, 191.0 / 255.0)
SUPPORTED_SCHEMES = ("Polar", "CC", "BCH")
RATES = {"Polar": POLAR_RATES, "CC": CC_RATES, "BCH": BCH_RATES}


def _rate_in(rate, allowed, tol=1e-12):
    return any(abs(float(rate) - float(value)) <= tol for value in allowed)


def crc11_append(payload):
    """Append the CRC-11 used by the existing Polar frame contract."""
    bits = np.asarray(payload, dtype=np.uint8)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Payload must be a one-dimensional binary array")
    reg = 0
    for bit in bits:
        feedback = ((reg >> 10) & 1) ^ int(bit)
        reg = (reg << 1) & 0x7FF
        if feedback:
            reg ^= CRC_POLYNOMIAL
    suffix = np.array([(reg >> j) & 1 for j in range(10, -1, -1)], dtype=np.uint8)
    return np.concatenate((bits, suffix))


def crc11_check(information):
    bits = np.asarray(information, dtype=np.uint8)
    if bits.size < CRC_BITS:
        return False
    reg = 0
    for bit in bits:
        feedback = ((reg >> 10) & 1) ^ int(bit)
        reg = (reg << 1) & 0x7FF
        if feedback:
            reg ^= CRC_POLYNOMIAL
    return reg == 0


def _uniform_keep(base_length, target_length, preserve_tail):
    """Return a deterministic keep mask with exactly ``target_length`` bits.

    The final ``preserve_tail`` coded bits (the zero-tail trellis outputs) are
    never punctured.  The remaining target positions are spread uniformly over
    the remaining base positions.  This is an explicit engineering pattern, not
    a standardized CCSDS puncturing pattern.
    """
    base_length = int(base_length)
    target_length = int(target_length)
    preserve_tail = int(preserve_tail)
    if not 0 < target_length <= base_length:
        raise ValueError("target_length must be in (0, base_length]")
    if preserve_tail < 0 or preserve_tail > target_length:
        raise ValueError("preserve_tail must be within target_length")
    head_base = base_length - preserve_tail
    head_keep = target_length - preserve_tail
    keep = np.zeros(base_length, dtype=bool)
    if head_keep:
        indices = np.linspace(0, head_base - 1, head_keep).round().astype(int)
        if len(np.unique(indices)) != head_keep:
            raise ValueError("Uniform puncturing produced duplicate positions")
        keep[indices] = True
    keep[head_base:] = True
    return keep


class PolarCodec:
    """Adapter exposing PolarCode through the common codec contract."""

    scheme = "Polar"

    def __init__(self, n=256, rate=0.5, crc_bits=CRC_BITS, list_size=8):
        if not _rate_in(rate, POLAR_RATES):
            raise ValueError(f"Unsupported Polar rate: {rate}")
        self.n = int(n)
        self.rate = float(rate)
        self.crc_bits = int(crc_bits)
        self.inner = PolarCode(n=self.n, rate=self.rate,
                               crc_bits=self.crc_bits, list_size=list_size)
        self.coded_bits = self.n
        self.information_bits = self.inner.k
        self.payload_bits = self.inner.payload_bits
        self.actual_rate = self.inner.actual_rate
        self.payload_rate = self.inner.payload_rate

    def encode(self, payload):
        return self.inner.encode(payload)

    def decode_with_status(self, llr):
        return self.inner.decode_with_status(llr)


class ConvolutionalCode:
    """K=7, rate-1/2 mother code with zero termination.

    Generator polynomials are (171,133)_8.  A single input bit produces two
    output bits; the trellis starts and ends in state zero.  The common CRC-11
    is prepended to the six zero tail bits, so ``payload_bits`` excludes both
    CRC and termination.  Rates 2/3 and 3/4 are obtained by the deterministic
    puncturing mask described in ``_uniform_keep``.
    """

    scheme = "CC"
    CONSTRAINT_LENGTH = 7
    TAIL_BITS = CONSTRAINT_LENGTH - 1
    GENERATORS = (0o171, 0o133)

    def __init__(self, n=256, rate=0.5, crc_bits=CRC_BITS):
        if not _rate_in(rate, CC_RATES):
            raise ValueError(f"Unsupported CC rate: {rate}")
        self.n = int(n)
        if self.n < 32:
            raise ValueError("CC coded length must be >= 32")
        self.rate = float(rate)
        self.crc_bits = int(crc_bits)
        self.coded_bits = self.n
        self.k_encoder = int(math.ceil(self.n * self.rate))
        self.information_bits = self.k_encoder - self.TAIL_BITS
        self.payload_bits = self.information_bits - self.crc_bits
        if self.payload_bits <= 0:
            raise ValueError("CC rate/block length leaves no payload")
        self.actual_rate = self.information_bits / self.n
        self.payload_rate = self.payload_bits / self.n
        self.keep_mask = _uniform_keep(
            2 * self.k_encoder, self.n, preserve_tail=2 * self.TAIL_BITS)
        self.base_length = 2 * self.k_encoder

    @staticmethod
    def _parity(value):
        return bin(int(value)).count("1") & 1

    def _encode_base(self, bits):
        state = 0
        output = np.empty(2 * len(bits), dtype=np.uint8)
        for index, bit in enumerate(bits):
            window = (int(bit) << (self.CONSTRAINT_LENGTH - 1)) | state
            output[2 * index] = self._parity(window & self.GENERATORS[0])
            output[2 * index + 1] = self._parity(window & self.GENERATORS[1])
            # Keep the most recent input in bit 5.  The previous left-shift
            # update put u[t-1] in bit 0, so bit 5 of the next window was
            # u[t-6] rather than u[t-1]; that silently changed the claimed
            # (171,133) generator taps.
            state = ((state >> 1) | (int(bit) << (self.CONSTRAINT_LENGTH - 2))) & (
                (1 << (self.CONSTRAINT_LENGTH - 1)) - 1)
        return output

    def encode(self, payload):
        bits = np.asarray(payload, dtype=np.uint8)
        if bits.ndim != 1 or bits.size != self.payload_bits:
            raise ValueError(f"Expected {self.payload_bits} payload bits")
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Payload must be binary")
        information = crc11_append(bits) if self.crc_bits else bits.copy()
        terminated = np.concatenate(
            (information, np.zeros(self.TAIL_BITS, dtype=np.uint8)))
        return self._encode_base(terminated)[self.keep_mask]

    def _decode_base(self, llr):
        """Soft Viterbi on the rate-1/2 mother trellis."""
        metrics = np.full(1 << (self.CONSTRAINT_LENGTH - 1), -np.inf)
        metrics[0] = 0.0
        back_state = np.empty((self.k_encoder, metrics.size), dtype=np.int16)
        back_bit = np.empty((self.k_encoder, metrics.size), dtype=np.uint8)
        states = np.arange(metrics.size)
        for t in range(self.k_encoder):
            new_metrics = np.full_like(metrics, -np.inf)
            for state in states:
                if not np.isfinite(metrics[state]):
                    continue
                for bit in (0, 1):
                    next_state = ((int(state) >> 1) | (bit << (self.CONSTRAINT_LENGTH - 2))) & (metrics.size - 1)
                    window = (bit << (self.CONSTRAINT_LENGTH - 1)) | int(state)
                    expected0 = self._parity(window & self.GENERATORS[0])
                    expected1 = self._parity(window & self.GENERATORS[1])
                    branch = 0.5 * (
                        (1 - 2 * expected0) * llr[2 * t]
                        + (1 - 2 * expected1) * llr[2 * t + 1])
                    score = metrics[state] + branch
                    if score > new_metrics[next_state]:
                        new_metrics[next_state] = score
                        back_state[t, next_state] = state
                        back_bit[t, next_state] = bit
            metrics = new_metrics
        # The encoder appends six zero tail bits, so a valid terminated path
        # must finish in state zero.  Selecting the best of all final states
        # discarded that known constraint and was not a decoder for the
        # terminated code described above.
        state = 0 if np.isfinite(metrics[0]) else int(np.argmax(metrics))
        bits = np.empty(self.k_encoder, dtype=np.uint8)
        for t in range(self.k_encoder - 1, -1, -1):
            bits[t] = back_bit[t, state]
            state = int(back_state[t, state])
        return bits

    def decode_with_status(self, llr):
        values = np.asarray(llr, dtype=np.float64)
        if values.ndim != 1 or values.size != self.n or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.n} finite LLRs")
        base_llr = np.zeros(self.base_length)
        base_llr[self.keep_mask] = values
        bits = self._decode_base(base_llr)
        information = bits[: self.information_bits]
        payload = information[: self.payload_bits].copy()
        passed = bool(crc11_check(information)) if self.crc_bits else True
        return payload, passed


# --- Binary BCH(255,k), extended to 256 ---------------------------------
#
# GF(2^8) with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D).  Generator
# polynomials are built from the cyclotomic cosets of 1,3,...,2t-1.  The
# decoder is hard-decision Berlekamp-Massey + Chien search.  The extension bit
# is the overall parity bit; all coding modes keep the common CRC-11.
_BCH_PRIMITIVE = 0x11D
_BCH_EXP = [0] * 510
_BCH_LOG = [-1] * 256
_x = 1
for _i in range(255):
    _BCH_EXP[_i] = _x
    _BCH_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= _BCH_PRIMITIVE
for _i in range(255, 510):
    _BCH_EXP[_i] = _BCH_EXP[_i - 255]


def _gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _BCH_EXP[(_BCH_LOG[a] + _BCH_LOG[b]) % 255]


def _gf_inv(a):
    if a == 0:
        raise ZeroDivisionError("GF inverse of zero")
    return _BCH_EXP[(255 - _BCH_LOG[a]) % 255]


def _gf_div(a, b):
    return 0 if a == 0 else _BCH_EXP[(_BCH_LOG[a] - _BCH_LOG[b]) % 255]


def _poly_mul_gf(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] ^= _gf_mul(x, y)
    return out


def _minimal_polynomial(exponent):
    coset = []
    value = exponent
    while value not in coset:
        coset.append(value)
        value = (2 * value) % 255
    polynomial = [1]
    for root in coset:
        polynomial = _poly_mul_gf(polynomial, [_BCH_EXP[root], 1])
    if not all(coefficient in (0, 1) for coefficient in polynomial):
        raise RuntimeError("BCH minimal polynomial did not reduce to binary coefficients")
    return [int(coefficient) for coefficient in polynomial]


@functools.lru_cache(maxsize=None)
def _bch_generator(t):
    """Generator polynomial for consecutive roots alpha^1..alpha^(2t-1)."""
    generator = [1]
    for exponent in range(1, 2 * t, 2):
        generator = _poly_mul_gf(generator, _minimal_polynomial(exponent))
    return tuple(int(coefficient) for coefficient in generator)


def _bch_syndrome(bits, exponent):
    alpha = _BCH_EXP[exponent % 255]
    value = 0
    for bit in bits:
        value = _gf_mul(value, alpha) ^ int(bit)
    return value


def _bch_berlekamp_massey(syndromes, t):
    """Return the error-locator polynomial, low-degree coefficient first."""
    locator = [1]
    previous = [1]
    length = 0
    shift = 1
    previous_discrepancy = 1
    for n in range(1, 2 * t + 1):
        discrepancy = syndromes[n - 1]
        for i in range(1, length + 1):
            if i < len(locator):
                discrepancy ^= _gf_mul(locator[i], syndromes[n - 1 - i])
        if discrepancy == 0:
            shift += 1
        elif 2 * length <= n - 1:
            old = locator[:]
            scale = _gf_div(discrepancy, previous_discrepancy)
            required = len(previous) + shift
            if len(locator) < required:
                locator += [0] * (required - len(locator))
            for i, value in enumerate(previous):
                locator[i + shift] ^= _gf_mul(scale, value)
            length = n - length
            previous = old
            previous_discrepancy = discrepancy
            shift = 1
        else:
            scale = _gf_div(discrepancy, previous_discrepancy)
            required = len(previous) + shift
            if len(locator) < required:
                locator += [0] * (required - len(locator))
            for i, value in enumerate(previous):
                locator[i + shift] ^= _gf_mul(scale, value)
            shift += 1
    return locator[: length + 1]


def _bch_error_positions(locator):
    """Chien search; map field exponents to bit positions (left-to-right)."""
    positions = []
    for exponent in range(255):
        root = _gf_inv(_BCH_EXP[exponent % 255]) if exponent % 255 else 1
        value = 0
        power = 1
        for coefficient in locator:
            value ^= _gf_mul(coefficient, power)
            power = _gf_mul(power, root)
        if value == 0:
            positions.append(254 - exponent)
    return positions


def _bch_encode_message(message, generator_low_to_high):
    """Systematic encoding of a 255-bit BCH message; returns 255 bits."""
    k = len(message)
    parity_count = 255 - k
    work = [int(bit) for bit in message] + [0] * parity_count
    generator_high_to_low = list(reversed(generator_low_to_high))
    for i in range(k):
        if work[i]:
            for j, coefficient in enumerate(generator_high_to_low):
                work[i + j] ^= coefficient
    return np.array([int(bit) for bit in message] + work[k:], dtype=np.uint8)


class BCHCode:
    """Extended binary BCH(255,k)+parity, hard-decision baseline."""

    scheme = "BCH"

    def __init__(self, n=256, rate=0.5, crc_bits=CRC_BITS):
        if int(n) != 256:
            raise ValueError("BCH implementation is defined for the 256-bit frame")
        if not _rate_in(rate, BCH_RATES):
            raise ValueError(f"Unsupported BCH rate: {rate}")
        self.n = 256
        self.rate = float(rate)
        self.crc_bits = int(crc_bits)
        self.k_bch = int(round(self.rate * 255))
        self.information_bits = self.k_bch
        self.payload_bits = self.k_bch - self.crc_bits
        if self.payload_bits <= 0:
            raise ValueError("BCH rate leaves no payload")
        self.coded_bits = 256
        self.actual_rate = self.k_bch / 256.0
        self.payload_rate = self.payload_bits / 256.0
        self.t = None
        self.generator = None
        for trial_t in range(1, 32):
            generator = _bch_generator(trial_t)
            if 255 - (len(generator) - 1) == self.k_bch:
                self.t = trial_t
                self.generator = generator
                break
        if self.t is None:
            raise ValueError(f"No BCH generator found for k={self.k_bch}")

    def encode(self, payload):
        bits = np.asarray(payload, dtype=np.uint8)
        if bits.ndim != 1 or bits.size != self.payload_bits:
            raise ValueError(f"Expected {self.payload_bits} payload bits")
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Payload must be binary")
        information = crc11_append(bits) if self.crc_bits else bits.copy()
        encoded = _bch_encode_message(information, self.generator)
        extension = np.array([int(np.bitwise_xor.reduce(encoded))], dtype=np.uint8)
        return np.concatenate((encoded, extension))

    def _decode_base(self, received):
        syndromes = [_bch_syndrome(received, j) for j in range(1, 2 * self.t + 1)]
        if not any(syndromes):
            return received.copy(), 0, True
        locator = _bch_berlekamp_massey(syndromes, self.t)
        positions = _bch_error_positions(locator)
        if len(positions) != len(locator) - 1:
            return received.copy(), 0, False
        corrected = received.copy()
        for position in positions:
            if not 0 <= position < 255:
                return received.copy(), 0, False
            corrected[position] ^= 1
        check = [_bch_syndrome(corrected, j) for j in range(1, 2 * self.t + 1)]
        return corrected, len(positions), not any(check)

    def decode_with_status(self, llr):
        values = np.asarray(llr, dtype=np.float64)
        if values.ndim != 1 or values.size != self.n or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.n} finite LLRs")
        hard = (values < 0).astype(np.uint8)
        corrected, corrected_count, base_ok = self._decode_base(hard[:255])
        information = corrected[: self.k_bch]
        payload = information[: self.payload_bits].copy()
        # Use the extension bit as the overall-parity check of the extended
        # BCH code.  A parity mismatch after a base correction means one more
        # error than the base decoder corrected: it is correctable only while
        # the total corrected weight stays <= t.  If the base decoder already
        # used all t corrections, the mismatch is an uncorrectable (t+1)-error
        # indication and the frame must be rejected.  This also keeps a single
        # extension-bit error correctable, as required by the extended code.
        parity = int(np.bitwise_xor.reduce(corrected))
        extension_ok = (not base_ok) or (int(hard[255]) == parity) or (corrected_count < self.t)
        passed = bool(base_ok and extension_ok and
                      (crc11_check(information) if self.crc_bits else True))
        return payload, passed


def build_codec(scheme, n=256, rate=0.5, crc_bits=CRC_BITS, list_size=8):
    scheme = str(scheme)
    if scheme == "Polar":
        return PolarCodec(n=n, rate=rate, crc_bits=crc_bits, list_size=list_size)
    if scheme == "CC":
        return ConvolutionalCode(n=n, rate=rate, crc_bits=crc_bits)
    if scheme == "BCH":
        return BCHCode(n=n, rate=rate, crc_bits=crc_bits)
    raise ValueError(f"Unsupported coding scheme: {scheme}")


def payload_rate_for(scheme, n, rate, crc_bits=CRC_BITS, list_size=8):
    return build_codec(scheme, n=n, rate=rate, crc_bits=crc_bits, list_size=list_size).payload_rate

"""Polar FEC with polarization-weight construction and CRC-aided SCL decoding.

This is a standalone simulation code, not a claim of 3GPP/5G compliance.
The requested rate counts information plus CRC bits; payload rate is lower.
"""

from __future__ import annotations

import numpy as np


def polar_transform(bits: np.ndarray) -> np.ndarray:
    """Apply F^(tensor log2(N)); this convention needs no bit reversal."""
    x = np.asarray(bits, dtype=np.uint8).copy()
    n = x.size
    if x.ndim != 1 or n < 2 or n & (n - 1):
        raise ValueError("Polar block length must be a power of two >= 2")
    step = 1
    while step < n:
        blocks = x.reshape(-1, 2 * step)
        blocks[:, :step] ^= blocks[:, step:]
        step *= 2
    return x


class PolarCode:
    """Binary Polar code with CRC-11 x^11+x^10+x^9+x^5+1.

    Frozen channels use PW beta=2**(1/4), a deterministic approximate
    construction. All-zero frozen bits, no puncturing or rate matching.
    Positive input LLR means bit zero. ``k`` includes CRC bits.
    """

    def __init__(self, n=256, rate=0.5, crc_bits=11, list_size=8):
        self.n = int(n)
        if self.n != n or self.n < 2 or self.n & (self.n - 1):
            raise ValueError("n must be a power of two >= 2")
        if not np.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("rate must lie in (0, 1]")
        self.k = int(round(self.n * float(rate)))
        self.crc_bits = int(crc_bits)
        if self.crc_bits not in (0, 11):
            raise ValueError("Only CRC-11 or crc_bits=0 is implemented")
        self.list_size = int(list_size)
        if self.list_size != list_size or self.list_size < 1:
            raise ValueError("list_size must be a positive integer")
        self.payload_bits = self.k - self.crc_bits
        if self.payload_bits <= 0:
            raise ValueError("rate/block length must leave payload bits after CRC")
        self.actual_rate = self.k / self.n
        self.payload_rate = self.payload_bits / self.n
        ids = np.arange(self.n)
        score = np.zeros(self.n)
        beta = 2.0 ** 0.25
        for j in range(self.n.bit_length() - 1):
            score += ((ids >> j) & 1) * beta**j
        reliable = np.argsort(score, kind="stable")[-self.k :]
        self.info_indices = np.sort(reliable)
        self.frozen = np.ones(self.n, dtype=bool)
        self.frozen[self.info_indices] = False
        self._frozen_prefix = np.concatenate(([0], np.cumsum(~self.frozen)))

    @staticmethod
    def _validate_bits(bits, length):
        value = np.asarray(bits)
        if value.ndim != 1 or value.size != length:
            raise ValueError(f"Expected {length} bits in one dimension")
        if not np.all((value == 0) | (value == 1)):
            raise ValueError("Bits must be binary")
        return value.astype(np.uint8)

    @staticmethod
    def _crc_remainder(bits):
        reg = 0
        for bit in bits:
            feedback = ((reg >> 10) & 1) ^ int(bit)
            reg = (reg << 1) & 0x7FF
            if feedback:
                reg ^= 0x621
        return reg

    def add_crc(self, payload):
        bits = self._validate_bits(payload, self.payload_bits)
        if not self.crc_bits:
            return bits.copy()
        crc = self._crc_remainder(bits)
        suffix = np.array([(crc >> j) & 1 for j in range(10, -1, -1)], dtype=np.uint8)
        return np.concatenate((bits, suffix))

    def check_crc(self, information):
        bits = self._validate_bits(information, self.k)
        return not self.crc_bits or self._crc_remainder(bits) == 0

    def encode(self, payload):
        u = np.zeros(self.n, dtype=np.uint8)
        u[self.info_indices] = self.add_crc(payload)
        return polar_transform(u)

    def decode(self, llr):
        return self.decode_with_status(llr)[0]

    def decode_with_status(self, llr):
        """Return (payload, CRC_passed), falling back to best metric on failure."""
        values = np.asarray(llr, dtype=np.float64)
        if values.ndim != 1 or values.size != self.n or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.n} finite LLRs")
        alpha = np.clip(values, -1e6, 1e6)[None, :]
        decisions = np.zeros((1, self.n), dtype=np.uint8)
        metrics = np.zeros(1)
        _, decisions, metrics, _ = self._decode_node(alpha, decisions, metrics, 0)
        order = np.argsort(metrics, kind="stable")
        selected = int(order[0])
        passed = False
        for row in order:
            candidate = decisions[row, self.info_indices]
            if self.check_crc(candidate):
                selected = int(row)
                passed = True
                break
        information = decisions[selected, self.info_indices]
        return information[: self.payload_bits].copy(), passed

    def _decode_node(self, alpha, decisions, metrics, start):
        count, width = alpha.shape
        origin = np.arange(count)
        if self._frozen_prefix[start + width] == self._frozen_prefix[start]:
            metrics = metrics + np.logaddexp(0.0, -alpha).sum(axis=1)
            return np.zeros_like(alpha, dtype=np.uint8), decisions, metrics, origin
        if width == 1:
            candidates = np.concatenate(
                (metrics + np.logaddexp(0.0, -alpha[:, 0]),
                 metrics + np.logaddexp(0.0, alpha[:, 0]))
            )
            keep = np.argsort(candidates, kind="stable")[: self.list_size]
            source = keep % count
            bits = (keep // count).astype(np.uint8)
            decisions = decisions[source].copy()
            decisions[:, start] = bits
            return bits[:, None], decisions, candidates[keep], source
        half = width // 2
        a, b = alpha[:, :half], alpha[:, half:]
        left_llr = np.logaddexp(0.0, a + b) - np.logaddexp(a, b)
        left, decisions, metrics, left_source = self._decode_node(
            left_llr, decisions, metrics, start
        )
        a, b = a[left_source], b[left_source]
        right_llr = b + (1.0 - 2.0 * left) * a
        right, decisions, metrics, right_source = self._decode_node(
            right_llr, decisions, metrics, start + half
        )
        left = left[right_source]
        combined = np.concatenate((left ^ right, right), axis=1)
        return combined, decisions, metrics, left_source[right_source]

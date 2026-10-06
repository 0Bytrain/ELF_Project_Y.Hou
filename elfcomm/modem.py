"""Unit-energy coherent modems: ideal timing, phase and flat-channel CSI.

This matched-filter-equivalent link is not an RF or synchronization simulator.
Positive LLR means bit zero. MSK uses unprecoded binary CPFSK, h=1/2, and
exact log-domain BCJR on four phase states with known initial phase.
"""

from __future__ import annotations
import numpy as np

MODEM_VERSION = "coherent-awgn-msk-bcjr-v1"
SUPPORTED_MODULATIONS = ("BPSK", "QPSK", "MSK")
MSK_SAMPLES_PER_SYMBOL = 8


def bits_per_symbol(modulation: str) -> int:
    name = modulation.upper()
    if name not in SUPPORTED_MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return 2 if name == "QPSK" else 1


def _bits(values):
    b = np.asarray(values)
    if b.ndim != 1 or not np.all((b == 0) | (b == 1)):
        raise ValueError("Bits must be a one-dimensional binary array")
    return b.astype(np.uint8, copy=False)


def modulate_bits(bits, modulation):
    """Es=1. QPSK Gray maps I/Q signs; MSK outputs 8 samples per bit."""
    b = _bits(bits)
    name = modulation.upper()
    bits_per_symbol(name)
    signs = 1.0 - 2.0 * b.astype(float)
    if name == "BPSK":
        return signs.astype(np.complex128)
    if name == "QPSK":
        if len(signs) % 2:
            signs = np.append(signs, 1.0)
        return (signs[0::2] + 1j * signs[1::2]) / np.sqrt(2.0)
    if not len(b):
        return np.empty(0, dtype=np.complex128)
    starts = np.concatenate(([0.0], np.cumsum(signs[:-1]))) * np.pi / 2.0
    time = (np.arange(MSK_SAMPLES_PER_SYMBOL) + 0.5) / MSK_SAMPLES_PER_SYMBOL
    phase = starts[:, None] + signs[:, None] * np.pi / 2.0 * time
    return (np.exp(1j * phase) / np.sqrt(MSK_SAMPLES_PER_SYMBOL)).reshape(-1)


def _msk_llr(received, noise_density):
    sps = MSK_SAMPLES_PER_SYMBOL
    if len(received) % sps:
        raise ValueError("MSK sample count must be divisible by samples per bit")
    n = len(received) // sps
    if not n:
        return np.empty(0)
    time = (np.arange(sps) + 0.5) / sps
    states = np.arange(4)
    signs = np.array([1, -1])
    destinations = (states[:, None] + signs[None, :]) % 4
    branches = np.exp(1j * (states[:, None, None] * np.pi / 2.0
        + signs[None, :, None] * np.pi / 2.0 * time)) / np.sqrt(sps)
    metric = (2.0 / noise_density) * np.real(np.einsum(
        "tk,sbk->tsb", received.reshape(n, sps), branches.conj()))
    alpha = np.full((n + 1, 4), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(n):
        score = alpha[t, :, None] + metric[t]
        alpha[t + 1] = np.logaddexp(np.roll(score[:, 0], 1), np.roll(score[:, 1], -1))
        alpha[t + 1] -= np.max(alpha[t + 1])
    beta = np.zeros(4)
    llr = np.empty(n)
    for t in range(n - 1, -1, -1):
        future = metric[t] + beta[destinations]
        score = alpha[t, :, None] + future
        llr[t] = np.logaddexp.reduce(score[:, 0]) - np.logaddexp.reduce(score[:, 1])
        beta = np.logaddexp(future[:, 0], future[:, 1])
        beta -= np.max(beta)
    return llr


def demodulate_llr(received, modulation, noise_density, n_bits=None):
    """Complex noise variance E|n|^2=N0; output log P(0|y)/P(1|y)."""
    samples = np.asarray(received, dtype=np.complex128)
    if samples.ndim != 1 or not np.isfinite(noise_density) or noise_density <= 0:
        raise ValueError("Need one-dimensional samples and positive finite N0")
    name = modulation.upper()
    bits_per_symbol(name)
    if name == "BPSK":
        llr = 4.0 * samples.real / noise_density
    elif name == "QPSK":
        llr = np.empty(2 * len(samples))
        llr[0::2] = 2.0 * np.sqrt(2.0) * samples.real / noise_density
        llr[1::2] = 2.0 * np.sqrt(2.0) * samples.imag / noise_density
    else:
        llr = _msk_llr(samples, noise_density)
    if n_bits is not None:
        if not 0 <= n_bits <= len(llr):
            raise ValueError("n_bits inconsistent with received samples")
        llr = llr[:n_bits]
    return llr


def transmit_llr(coded_bits, modulation, esn0_db, rng):
    """Transmit through AWGN. Es/N0 is per QPSK symbol or BPSK/MSK bit."""
    bits = _bits(coded_bits)
    if not np.isfinite(esn0_db):
        raise ValueError("esn0_db must be finite")
    n0 = float(10.0 ** (-esn0_db / 10.0))
    signal = modulate_bits(bits, modulation)
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(len(signal))
                                + 1j * rng.standard_normal(len(signal)))
    return demodulate_llr(signal + noise, modulation, n0, len(bits))

"""Recording quality and added noise (Sec. 3.1.1, 3.3.2, Appendix D.1).

The SNR follows scikit-maad's temporal SNR: the waveform is split into
non-overlapping frames of N_t samples; L_i = 10 log10(max_n |x[n]|^2) per
frame; E_total is the dB value of the mean linear frame power and
E_background the median of the L_i; SNR = E_total - E_background.
"""

from __future__ import annotations

import numpy as np


def snr_db(x: np.ndarray, frame_samples: int = 512) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x) // frame_samples
    if n == 0:
        raise ValueError("recording shorter than one SNR frame")
    A = np.abs(x[: n * frame_samples]).reshape(n, frame_samples).max(axis=1)
    L = 10 * np.log10(np.maximum(A ** 2, 1e-30))
    e_total = 10 * np.log10(np.mean(10 ** (L / 10)))
    e_background = np.median(L)
    return float(e_total - e_background)


def add_noise_to_range(x: np.ndarray, low_db: float | None, high_db: float,
                       rng: np.random.Generator, frame_samples: int = 512,
                       max_iter: int = 60) -> tuple[np.ndarray, float, float]:
    """Add Gaussian noise whose standard deviation is found by bisection so that
    the measured SNR falls within [low_db, high_db] (low_db=None: no lower bound).

    Returns (noisy waveform, achieved SNR, noise standard deviation).
    """
    x = np.asarray(x, dtype=float)
    noise = rng.standard_normal(len(x))
    low = -np.inf if low_db is None else low_db
    target = high_db - 0.5 if low_db is None else 0.5 * (low + high_db)

    def measure(sigma: float) -> float:
        return snr_db(x + sigma * noise, frame_samples)

    s0 = measure(0.0)
    if low <= s0 <= high_db:
        return x.copy(), s0, 0.0

    lo, hi = 0.0, float(np.std(x)) or 1e-3
    while measure(hi) > target:          # SNR falls as the noise level grows
        hi *= 2.0
    sigma, snr = hi, measure(hi)
    for _ in range(max_iter):
        sigma = 0.5 * (lo + hi)
        snr = measure(sigma)
        if low <= snr <= high_db:
            break
        if snr > high_db:
            lo = sigma
        else:
            hi = sigma
    if not (low <= snr <= high_db):
        raise RuntimeError(f"bisection did not reach [{low}, {high_db}] dB (last {snr:.2f} dB)")
    return x + sigma * noise, snr, sigma

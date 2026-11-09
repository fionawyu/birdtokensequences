#!/usr/bin/env python3
"""Spectral-subtraction denoiser, built to reuse snr_proxy.py's own
active/inactive frame split (same STFT window, same band, same hysteresis
thresholds) rather than inventing a separate noise-detection convention.
No denoising code previously existed in this repo -- the "denoised" PNGs
elsewhere are spectrogram visualizations after a bandpass filter, not
audio-domain noise reduction.

Method (classic spectral subtraction, Boll 1979):
  1. STFT with the exact same nperseg/noverlap as snr_proxy.snr_db.
  2. Reuse snr_proxy's hysteresis(...) on the same per-frame band-limited
     95th-percentile amplitude to find INACTIVE frames.
  3. Noise magnitude profile = mean |STFT| over inactive frames (full
     spectrum, not band-limited, so subtraction reduces noise outside
     [1000,11000]Hz too).
  4. Oversubtract with a spectral floor to avoid musical-noise artifacts:
     |Z_denoised| = max(|Z| - alpha*noise_profile, floor*|Z|)
  5. Keep original phase, inverse STFT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft, istft

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from snr_proxy import load_audio, snr_db, hysteresis, SR, WIN_MS, BAND, TH_HI, TH_LO  # noqa: E402


def denoise_spectral_subtraction(y: np.ndarray, sr: int = SR, oversubtract: float = 1.5,
                                  floor: float = 0.05):
    nper = int(round(sr * WIN_MS / 1000.0))
    f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nper // 2)
    S = np.abs(Z)
    phase = np.angle(Z)

    band = (f >= BAND[0]) & (f <= BAND[1])
    S_band = S[band]
    if S_band.shape[1] < 4:
        return y.copy(), 0.0, False  # too short to denoise meaningfully

    score = np.percentile(S_band, 95, axis=0)
    rng = score.max() - score.min()
    norm = (score - score.min()) / rng if rng > 0 else np.zeros_like(score)
    active = hysteresis(norm, TH_HI, TH_LO)
    inactive = ~active

    fallback_used = False
    if inactive.sum() < max(4, 0.05 * len(inactive)):
        # Hysteresis found (almost) no silence -- typically the noisiest recordings,
        # where there's no quiet reference to profile in the first place. Fall back to
        # a percentile-based noise floor (lowest-energy 20% of frames, whatever they
        # are) instead of skipping denoising entirely.
        energy = (S_band ** 2).sum(axis=0)
        thresh = np.percentile(energy, 20)
        inactive = energy <= thresh
        fallback_used = True

    noise_profile = S[:, inactive].mean(axis=1, keepdims=True)  # (freq, 1)
    S_denoised = np.maximum(S - oversubtract * noise_profile, floor * S)
    Z_denoised = S_denoised * np.exp(1j * phase)

    _, y_denoised = istft(Z_denoised, fs=sr, nperseg=nper, noverlap=nper // 2)
    if len(y_denoised) > len(y):
        y_denoised = y_denoised[: len(y)]
    elif len(y_denoised) < len(y):
        y_denoised = np.pad(y_denoised, (0, len(y) - len(y_denoised)))
    return y_denoised, float(inactive.mean()), fallback_used


def main():
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-wav", required=True)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--oversubtract", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=0.05)
    args = ap.parse_args()

    y = load_audio(args.in_wav, sr=SR)
    natural_db, _ = snr_db(y)
    y_dn, inactive_frac, fallback_used = denoise_spectral_subtraction(y, oversubtract=args.oversubtract, floor=args.floor)
    denoised_db, _ = snr_db(y_dn)

    peak = np.max(np.abs(y_dn)) + 1e-12
    if peak > 0.999:
        y_dn = y_dn / peak * 0.999
    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, y_dn.astype(np.float32), SR)
    print(f"{args.in_wav}\tnatural={natural_db:.2f}\tdenoised={denoised_db:.2f}\t"
          f"delta={denoised_db - natural_db:+.2f}\tinactive_frac={inactive_frac:.2f}\tfallback={fallback_used}")


if __name__ == "__main__":
    main()

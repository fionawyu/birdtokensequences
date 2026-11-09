#!/usr/bin/env python3
"""Add calibrated white Gaussian noise to a recording so its *measured*
snr_proxy.snr_db() value hits a target dB level.

Calibration is done by bisection search on the added-noise gain, not a
closed-form SNR formula, because snr_proxy's metric is a P90/P10 in-band
frame-energy ratio (not a simple signal/noise power ratio) -- searching
directly on the actual metric guarantees the achieved value matches what
every other analysis in this repo means by "SNR", at the cost of a few
extra snr_db() evaluations per recording (each one is a cheap STFT on a
short clip).

If the recording's natural SNR is already at or below the target, no
noise is added and this is reported explicitly (achieved_db == natural_db,
note="target already at/below natural SNR, no noise added") rather than
silently forcing an unreachable target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from snr_proxy import load_audio, snr_db, SR  # noqa: E402


def inject_noise_to_target(y: np.ndarray, target_db: float, rng: np.random.Generator,
                            tol_db: float = 0.3, max_iter: int = 40):
    """Bisection-search a Gaussian noise gain so snr_db(y + gain*noise) ~= target_db.

    Returns (mixed_audio, achieved_db, gain_used, note).
    """
    natural_db, _ = snr_db(y)
    if natural_db <= target_db:
        return y.copy(), float(natural_db), 0.0, "target already at/below natural SNR, no noise added"

    noise = rng.standard_normal(len(y)).astype(np.float64)
    sig_scale = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12

    lo, hi = 0.0, sig_scale * 10.0
    # expand hi until we overshoot below target (guards against a bad initial bracket)
    for _ in range(20):
        mixed = y + hi * noise
        db_hi, _ = snr_db(mixed)
        if db_hi <= target_db:
            break
        hi *= 2.0
    else:
        db_hi, _ = snr_db(y + hi * noise)

    best_gain, best_db, best_mixed = hi, db_hi, y + hi * noise
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mixed = y + mid * noise
        db_mid, _ = snr_db(mixed)
        if abs(db_mid - target_db) < abs(best_db - target_db):
            best_gain, best_db, best_mixed = mid, db_mid, mixed
        if abs(db_mid - target_db) <= tol_db:
            best_gain, best_db, best_mixed = mid, db_mid, mixed
            break
        # snr decreases monotonically as gain increases
        if db_mid > target_db:
            lo = mid
        else:
            hi = mid

    note = "ok" if abs(best_db - target_db) <= tol_db else f"tolerance missed by {abs(best_db - target_db):.2f}dB"
    return best_mixed, float(best_db), float(best_gain), note


def main():
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-wav", required=True)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--target-db", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    y = load_audio(args.in_wav, sr=SR)
    rng = np.random.default_rng(args.seed)
    mixed, achieved_db, gain, note = inject_noise_to_target(y, args.target_db, rng)

    peak = np.max(np.abs(mixed)) + 1e-12
    if peak > 0.999:
        mixed = mixed / peak * 0.999

    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, mixed.astype(np.float32), SR)
    print(f"{args.in_wav}\ttarget={args.target_db}\tachieved={achieved_db:.2f}\tgain={gain:.4f}\t{note}")


if __name__ == "__main__":
    main()

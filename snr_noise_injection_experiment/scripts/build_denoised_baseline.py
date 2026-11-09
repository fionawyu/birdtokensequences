#!/usr/bin/env python3
"""Denoise all 65 recordings (spectral subtraction, with the percentile
fallback for recordings where the hysteresis segmenter finds no silence to
profile), verify every one clears 10 dB, and write the denoised WAVs to
$SCRATCH as the common clean baseline for the noise-gradient injection
that follows.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snr_proxy import load_audio, snr_db, SR  # noqa: E402
from denoise_recording import denoise_spectral_subtraction  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
DENOISED_AUDIO_DIR = Path("outputs/noise_injection_full/denoised_audio")
TARGET_FLOOR_DB = 10.0


def main():
    rows = list(csv.DictReader(open(EXP_DIR / "data" / "recordings_65.csv")))
    DENOISED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    out = []
    below_floor = []
    for r in rows:
        y = load_audio(r["file_path"], sr=SR)
        natural_db, _ = snr_db(y)
        y_dn, inactive_frac, fallback = denoise_spectral_subtraction(y)
        denoised_db, _ = snr_db(y_dn)

        out_path = DENOISED_AUDIO_DIR / r["species"] / f"{r['recording_name']}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        peak = np.max(np.abs(y_dn)) + 1e-12
        y_write = y_dn / peak * 0.999 if peak > 0.999 else y_dn
        sf.write(str(out_path), y_write.astype(np.float32), SR)

        if denoised_db < TARGET_FLOOR_DB:
            below_floor.append(r["recording_name"])

        out.append(dict(species=r["species"], recording_name=r["recording_name"],
                         file_path=r["file_path"], natural_snr_db=round(natural_db, 2),
                         denoised_snr_db=round(denoised_db, 2), fallback_used=fallback,
                         denoised_wav_path=str(out_path)))
        print(f"{r['species']:6s} {r['recording_name']:45s} natural={natural_db:6.2f} "
              f"denoised={denoised_db:6.2f} fallback={fallback}")

    out_path = EXP_DIR / "data" / "denoised_baseline_65.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    vals = [r["denoised_snr_db"] for r in out]
    print(f"\nN={len(out)}  min={min(vals):.2f}  median={sorted(vals)[len(vals)//2]:.2f}  max={max(vals):.2f}")
    print(f"Below {TARGET_FLOOR_DB} dB after denoising: {len(below_floor)} -> {below_floor}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

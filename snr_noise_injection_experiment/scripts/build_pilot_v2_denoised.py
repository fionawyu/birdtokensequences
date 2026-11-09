#!/usr/bin/env python3
"""Pilot v2: same 24-recording stratified subset as the first pilot, but
noise is now injected starting from the DENOISED audio (data/denoised_baseline_65.csv)
instead of the raw recordings, so every recording starts from a common
>=15.6dB clean baseline before degrading -- no more no-op conditions from
already-noisy source recordings.

Same 5 candidate targets (12, 9, 6, 3, 0 dB) plus the untouched-denoised
baseline as a 6th condition. The 0dB target is expected to floor near
~2.0dB again (inherent to the snr_proxy metric under white noise, not
related to the audio's starting point).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snr_proxy import load_audio, SR  # noqa: E402
from inject_noise import inject_noise_to_target  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
TARGETS_DB = [12, 9, 6, 3, 0]
NOISY_AUDIO_DIR = Path("outputs/noise_injection_pilot_v2/audio")

# Same 24 recordings selected in the first pilot (species, recording_name),
# for direct comparability -- read from the previous pilot's manifest.
PREV_MANIFEST = EXP_DIR / "manifests" / "pilot_manifest.tsv"


def main():
    prev_rows = list(csv.DictReader(open(PREV_MANIFEST), delimiter="\t"))
    selected_keys = sorted(set((r["species"], r["recording_name"]) for r in prev_rows))
    print(f"Reusing {len(selected_keys)} recordings from the first pilot")

    denoised_rows = {(r["species"], r["recording_name"]): r
                      for r in csv.DictReader(open(EXP_DIR / "data" / "denoised_baseline_65.csv"))}

    NOISY_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    calib_rows = []
    task_id = 0

    for species, rec_name in selected_keys:
        r = denoised_rows[(species, rec_name)]
        denoised_path = r["denoised_wav_path"]
        denoised_db = float(r["denoised_snr_db"])
        y = load_audio(denoised_path, sr=SR)

        task_id += 1
        manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                   condition="baseline", target_db="", achieved_db=f"{denoised_db:.2f}",
                                   wav_path=denoised_path))
        calib_rows.append(dict(species=species, recording_name=rec_name, condition="baseline",
                                denoised_db=f"{denoised_db:.2f}", target_db="", achieved_db=f"{denoised_db:.2f}",
                                gain="0", note="denoised, no additional noise added"))

        for target in TARGETS_DB:
            rng = np.random.default_rng(hash((rec_name, target, "v2")) % (2**32))
            mixed, achieved_db, gain, note = inject_noise_to_target(y, float(target), rng)
            out_path = NOISY_AUDIO_DIR / species / rec_name / f"snr_{target}db.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            import soundfile as sf
            peak = np.max(np.abs(mixed)) + 1e-12
            if peak > 0.999:
                mixed = mixed / peak * 0.999
            sf.write(str(out_path), mixed.astype(np.float32), SR)

            task_id += 1
            manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                       condition=f"snr_{target}db", target_db=target, achieved_db=f"{achieved_db:.2f}",
                                       wav_path=str(out_path)))
            calib_rows.append(dict(species=species, recording_name=rec_name, condition=f"snr_{target}db",
                                    denoised_db=f"{denoised_db:.2f}", target_db=target, achieved_db=f"{achieved_db:.2f}",
                                    gain=f"{gain:.4f}", note=note))
            print(f"{species:6s} {rec_name:45s} denoised={denoised_db:6.2f} target={target:3d} "
                  f"achieved={achieved_db:6.2f} note={note}")

    manifest_path = EXP_DIR / "manifests" / "pilot_v2_manifest.tsv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "species", "recording_name", "condition",
                                           "target_db", "achieved_db", "wav_path"], delimiter="\t",
                            lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nWrote {manifest_path} ({len(manifest_rows)} rows)")

    calib_path = EXP_DIR / "data" / "pilot_v2_noise_calibration.csv"
    with calib_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "recording_name", "condition", "denoised_db",
                                           "target_db", "achieved_db", "gain", "note"], lineterminator="\n")
        w.writeheader()
        w.writerows(calib_rows)
    print(f"Wrote {calib_path}")


if __name__ == "__main__":
    main()

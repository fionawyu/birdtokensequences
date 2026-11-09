#!/usr/bin/env python3
"""Build the full 65-recording x 6-condition manifest (390 rows), reusing
pilot v2's already-computed noisy audio + GP fits for its 24 recordings
(same denoised-baseline approach, same 5 noise targets) instead of
redoing them -- only the other 41 recordings need fresh noise injection.
The Slurm array's results-dir is deliberately the SAME as pilot v2's, so
run_consistency_recording.py's own DONE-marker skip logic reuses those 144
fits automatically; only ~246 new fits actually run.
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
PILOT_V2_AUDIO_DIR = Path("outputs/noise_injection_pilot_v2/audio")
FULL_NEW_AUDIO_DIR = Path("outputs/noise_injection_full/noisy_audio")


def main():
    denoised_rows = list(csv.DictReader(open(EXP_DIR / "data" / "denoised_baseline_65.csv")))
    pilot_keys = set((r["species"], r["recording_name"])
                      for r in csv.DictReader(open(EXP_DIR / "manifests" / "pilot_v2_manifest.tsv"), delimiter="\t"))
    print(f"Reusing {len(pilot_keys)} recordings' audio+fits from pilot v2; "
          f"generating fresh noise injection for {len(denoised_rows) - len(pilot_keys)} more")

    FULL_NEW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    calib_rows = []
    task_id = 0
    n_reused, n_fresh = 0, 0

    for r in denoised_rows:
        species, rec_name = r["species"], r["recording_name"]
        denoised_db = float(r["denoised_snr_db"])
        key = (species, rec_name)
        reused = key in pilot_keys

        task_id += 1
        # baseline condition always uses the denoised audio directly (same for both reused/fresh)
        manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                   condition="baseline", target_db="", achieved_db=f"{denoised_db:.2f}",
                                   wav_path=r["denoised_wav_path"], reused=reused))

        if reused:
            n_reused += 1
            for target in TARGETS_DB:
                task_id += 1
                noisy_path = PILOT_V2_AUDIO_DIR / species / rec_name / f"snr_{target}db.wav"
                assert noisy_path.exists(), f"expected pilot v2 audio missing: {noisy_path}"
                manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                           condition=f"snr_{target}db", target_db=target, achieved_db="",
                                           wav_path=str(noisy_path), reused=True))
        else:
            n_fresh += 1
            y = load_audio(r["denoised_wav_path"], sr=SR)
            for target in TARGETS_DB:
                rng = np.random.default_rng(hash((rec_name, target, "full")) % (2**32))
                mixed, achieved_db, gain, note = inject_noise_to_target(y, float(target), rng)
                out_path = FULL_NEW_AUDIO_DIR / species / rec_name / f"snr_{target}db.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                import soundfile as sf
                peak = np.max(np.abs(mixed)) + 1e-12
                if peak > 0.999:
                    mixed = mixed / peak * 0.999
                sf.write(str(out_path), mixed.astype(np.float32), SR)

                task_id += 1
                manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                           condition=f"snr_{target}db", target_db=target, achieved_db=f"{achieved_db:.2f}",
                                           wav_path=str(out_path), reused=False))
                calib_rows.append(dict(species=species, recording_name=rec_name, condition=f"snr_{target}db",
                                        denoised_db=f"{denoised_db:.2f}", target_db=target, achieved_db=f"{achieved_db:.2f}",
                                        gain=f"{gain:.4f}", note=note))
                print(f"{species:6s} {rec_name:45s} denoised={denoised_db:6.2f} target={target:3d} "
                      f"achieved={achieved_db:6.2f} note={note}")

    # backfill achieved_db for reused rows from pilot v2's own calibration log
    pilot_calib = {(r["species"], r["recording_name"], r["condition"]): r["achieved_db"]
                    for r in csv.DictReader(open(EXP_DIR / "data" / "pilot_v2_noise_calibration.csv"))}
    for row in manifest_rows:
        if row.get("reused") and row["condition"] != "baseline":
            key = (row["species"], row["recording_name"], row["condition"])
            row["achieved_db"] = pilot_calib[key]

    manifest_path = EXP_DIR / "manifests" / "full_run_manifest.tsv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "species", "recording_name", "condition",
                                           "target_db", "achieved_db", "wav_path", "reused"],
                            delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nWrote {manifest_path} ({len(manifest_rows)} rows; "
          f"{n_reused} recordings reused from pilot v2, {n_fresh} freshly injected)")

    if calib_rows:
        calib_path = EXP_DIR / "data" / "full_run_noise_calibration.csv"
        with calib_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["species", "recording_name", "condition", "denoised_db",
                                               "target_db", "achieved_db", "gain", "note"], lineterminator="\n")
            w.writeheader()
            w.writerows(calib_rows)
        print(f"Wrote {calib_path}")


if __name__ == "__main__":
    main()

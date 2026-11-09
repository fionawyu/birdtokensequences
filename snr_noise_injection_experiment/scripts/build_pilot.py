#!/usr/bin/env python3
"""Build the pilot: select a stratified subset of the 65-recording
four-species set, generate Gaussian-noise-injected copies at 5 candidate
target SNR levels (12, 9, 6, 3, 0 dB) plus the untouched baseline, and
write a Slurm-array manifest (one row per recording x condition fit task).

Selection: 6 recordings per species (24 total), chosen to span each
species' own natural-SNR range (min/max/and evenly spaced quantiles in
between) rather than randomly -- so the pilot's calibration check covers
both the cleanest and noisiest recordings in the set, not just typical
ones.

Run via srun (audio I/O + STFT for 24*5=120 injections is real compute,
not login-node work), NOT directly on the login node.
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
N_PER_SPECIES = 6
NOISY_AUDIO_DIR = Path("outputs/noise_injection_pilot/audio")


def select_stratified(rows, n):
    rows_sorted = sorted(rows, key=lambda r: float(r["natural_snr_db"]))
    if len(rows_sorted) <= n:
        return rows_sorted
    idx = np.linspace(0, len(rows_sorted) - 1, n)
    idx = sorted(set(int(round(i)) for i in idx))
    while len(idx) < n:
        for i in range(len(rows_sorted)):
            if i not in idx:
                idx.append(i)
                break
        idx = sorted(idx)
    return [rows_sorted[i] for i in idx[:n]]


def main():
    baseline_rows = list(csv.DictReader(open(EXP_DIR / "data" / "baseline_snr.csv")))
    by_species: dict[str, list] = {}
    for r in baseline_rows:
        by_species.setdefault(r["species"], []).append(r)

    selected = []
    for sp, rows in by_species.items():
        chosen = select_stratified(rows, N_PER_SPECIES)
        selected.extend(chosen)
    print(f"Selected {len(selected)} recordings ({', '.join(f'{sp}={len(select_stratified(by_species[sp], N_PER_SPECIES))}' for sp in by_species)})")

    NOISY_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    calib_rows = []
    task_id = 0

    for r in selected:
        species = r["species"]
        rec_name = r["recording_name"]
        src_path = REPO_ROOT / r["file_path"]
        natural_db = float(r["natural_snr_db"])
        y = load_audio(str(src_path), sr=SR)

        # baseline condition: untouched original audio, fit directly from source path
        task_id += 1
        manifest_rows.append(dict(task_id=task_id, species=species, recording_name=rec_name,
                                   condition="baseline", target_db="", achieved_db=f"{natural_db:.2f}",
                                   wav_path=str(src_path)))
        calib_rows.append(dict(species=species, recording_name=rec_name, condition="baseline",
                                natural_db=f"{natural_db:.2f}", target_db="", achieved_db=f"{natural_db:.2f}",
                                gain="0", note="original, no noise added"))

        for target in TARGETS_DB:
            rng = np.random.default_rng(hash((rec_name, target)) % (2**32))
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
                                    natural_db=f"{natural_db:.2f}", target_db=target, achieved_db=f"{achieved_db:.2f}",
                                    gain=f"{gain:.4f}", note=note))
            print(f"{species:6s} {rec_name:45s} natural={natural_db:6.2f} target={target:3d} "
                  f"achieved={achieved_db:6.2f} note={note}")

    manifest_path = EXP_DIR / "manifests" / "pilot_manifest.tsv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "species", "recording_name", "condition",
                                           "target_db", "achieved_db", "wav_path"], delimiter="\t",
                            lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nWrote {manifest_path} ({len(manifest_rows)} rows)")

    calib_path = EXP_DIR / "data" / "pilot_noise_calibration.csv"
    with calib_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "recording_name", "condition", "natural_db",
                                           "target_db", "achieved_db", "gain", "note"], lineterminator="\n")
        w.writeheader()
        w.writerows(calib_rows)
    print(f"Wrote {calib_path}")


if __name__ == "__main__":
    main()

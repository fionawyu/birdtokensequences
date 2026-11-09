#!/usr/bin/env python3
"""Per-recording SNR screening for the raw pool, using quick_screen.py's metric.

quick_screen.py itself only ever computed maad.sound.temporal_snr and
compared the *per-dataset median* against --snr-threshold. This script
reuses its exact load_audio()/calculate_temporal_snr() functions (imported,
not reimplemented) but applies the same >= 4.0 dB decision rule per
individual recording, since the screening_summary.csv this experiment needs
is per-recording. This is the SNR metric/threshold the user chose after the
audit surfaced that quick_screen.py and snr_proxy.py disagree.

Additive: does not modify quick_screen.py, does not touch results/quick_screen_*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quick_screen import load_audio, calculate_temporal_snr  # noqa: E402
from config import RESULTS_DIR, SNR_THRESHOLD_DB  # noqa: E402


def compute_snr_for_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for i, row in manifest.iterrows():
        path = row["raw_path"]
        try:
            y = load_audio(path)
            energy_db, background_db, snr_db = calculate_temporal_snr(y)
            status = "ok"
        except Exception as exc:
            energy_db = background_db = snr_db = float("nan")
            status = f"failed: {exc}"
        rows.append(
            {
                "recording_id": row["recording_id"],
                "species": row["species"],
                "snr_db": snr_db,
                "energy_db": energy_db,
                "background_db": background_db,
                "snr_status": status,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(manifest)}")
    snr_df = pd.DataFrame(rows)
    snr_df["snr_pass"] = snr_df["snr_db"] >= SNR_THRESHOLD_DB
    return snr_df


def main() -> None:
    manifest_path = RESULTS_DIR / "raw_pool_manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path}; run build_pool_manifest.py first")

    snr_df = compute_snr_for_manifest(manifest_path)
    out_path = RESULTS_DIR / "raw_pool_snr.csv"
    snr_df.to_csv(out_path, index=False)

    summary = snr_df.groupby("species").agg(
        n=("recording_id", "size"),
        n_pass=("snr_pass", "sum"),
        median_snr_db=("snr_db", "median"),
        n_failed_to_load=("snr_status", lambda s: (s != "ok").sum()),
    )
    print(summary)
    print(f"\nWrote {out_path} ({len(snr_df)} rows)")


if __name__ == "__main__":
    main()

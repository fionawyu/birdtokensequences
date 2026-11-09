#!/usr/bin/env python3
"""
Quick species-level audio screening.

Uses two established Python measures:
1. Recording quality: maad.sound.temporal_snr
2. Spectral character: librosa.feature.spectral_flatness

The retain/review decision is based ONLY on median temporal SNR.
Spectral flatness is descriptive and must not be used by itself to exclude a species.

Folder layout:
root/
    Chickadee/*.wav
    WTSP/*.wav
    YWAR/*.wav

Examples:
    python quick_species_screen.py --root /data/audio --sample 20 --snr-threshold 4
    python quick_species_screen.py --root /data/audio --sample 0 --snr-threshold 4
    python quick_species_screen.py --selftest

Outputs:
    recording_screen.csv
    species_screen.csv
    species_screen.md
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3", ".ogg", ".OGG")
DEFAULT_SR = 44100


def require_packages():
    missing = []
    try:
        import librosa  # noqa: F401
    except Exception:
        missing.append("librosa")
    try:
        from maad.sound import temporal_snr  # noqa: F401
    except Exception:
        missing.append("scikit-maad")
    if missing:
        sys.exit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with:\n    pip install librosa scikit-maad soundfile pandas scipy"
        )


def collect_files(root):
    root = Path(root)
    if not root.exists():
        sys.exit(f"root does not exist: {root}")

    items = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ext in AUDIO_EXTENSIONS:
            pattern = str(dataset_dir / "**" / f"*{ext}")
            for path in sorted(glob.glob(pattern, recursive=True)):
                items.append((dataset_dir.name, path))
    return items


def sample_per_dataset(items, n, seed):
    if n <= 0:
        return items

    rng = np.random.default_rng(seed)
    grouped = {}
    for dataset, path in items:
        grouped.setdefault(dataset, []).append(path)

    sampled = []
    for dataset in sorted(grouped):
        paths = grouped[dataset]
        idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        sampled.extend((dataset, paths[i]) for i in sorted(idx))
    return sampled


def load_audio(path, sr=DEFAULT_SR, max_seconds=0):
    import librosa

    duration = None if max_seconds <= 0 else float(max_seconds)
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        raise ValueError("empty audio")
    return y


def calculate_temporal_snr(y):
    from maad.sound import temporal_snr

    result = temporal_snr(y)
    if not isinstance(result, tuple) or len(result) < 3:
        raise RuntimeError(f"unexpected temporal_snr output: {result!r}")

    energy_db, background_db, snr_db = result[:3]
    return float(energy_db), float(background_db), float(snr_db)


def calculate_flatness(y, active_percentile=75.0):
    """Median spectral flatness over the recording's higher-energy frames."""
    import librosa

    n_fft = 1024
    hop_length = 512

    flatness = librosa.feature.spectral_flatness(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )[0]
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    n = min(len(flatness), len(rms))
    flatness = flatness[:n]
    rms = rms[:n]

    valid = np.isfinite(flatness) & np.isfinite(rms)
    flatness = flatness[valid]
    rms = rms[valid]
    if len(flatness) == 0:
        return np.nan, 0

    cutoff = np.percentile(rms, active_percentile)
    keep = rms >= cutoff
    if not np.any(keep):
        keep = np.ones(len(flatness), dtype=bool)

    return float(np.median(flatness[keep])), int(np.sum(keep))


def screen_recordings(items, sr, max_seconds, active_percentile):
    rows = []
    total = len(items)

    for i, (dataset, path) in enumerate(items, 1):
        try:
            y = load_audio(path, sr=sr, max_seconds=max_seconds)
            energy_db, background_db, snr_db = calculate_temporal_snr(y)
            flatness, n_flatness_frames = calculate_flatness(
                y,
                active_percentile=active_percentile,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "path": path,
                    "duration_s_used": len(y) / sr,
                    "energy_db": energy_db,
                    "background_db": background_db,
                    "temporal_snr_db": snr_db,
                    "spectral_flatness": flatness,
                    "flatness_frames_used": n_flatness_frames,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "path": path,
                    "duration_s_used": np.nan,
                    "energy_db": np.nan,
                    "background_db": np.nan,
                    "temporal_snr_db": np.nan,
                    "spectral_flatness": np.nan,
                    "flatness_frames_used": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        if i % 25 == 0 or i == total:
            print(f"[progress] {i}/{total}")

    return pd.DataFrame(rows)


def summarize_species(recordings, snr_threshold):
    good = recordings[
        (recordings["status"] == "ok")
        & np.isfinite(recordings["temporal_snr_db"])
        & np.isfinite(recordings["spectral_flatness"])
    ].copy()

    if good.empty:
        raise RuntimeError("no recordings produced valid SNR and flatness values")

    rows = []
    for dataset, group in good.groupby("dataset", sort=True):
        snr = group["temporal_snr_db"].to_numpy(float)
        flat = group["spectral_flatness"].to_numpy(float)

        snr_q1, snr_q3 = np.percentile(snr, [25, 75])
        flat_q1, flat_q3 = np.percentile(flat, [25, 75])
        median_snr = float(np.median(snr))

        decision = (
            "retain candidate: recording-quality threshold passed"
            if median_snr >= snr_threshold
            else "review/exclude candidate: low median temporal SNR"
        )

        rows.append(
            {
                "dataset": dataset,
                "n_recordings": len(group),
                "median_snr_db": median_snr,
                "snr_q1": float(snr_q1),
                "snr_q3": float(snr_q3),
                "median_spectral_flatness": float(np.median(flat)),
                "flatness_q1": float(flat_q1),
                "flatness_q3": float(flat_q3),
                "decision": decision,
            }
        )

    summary = pd.DataFrame(rows)
    summary["flatness_rank"] = summary["median_spectral_flatness"].rank(
        method="min", ascending=True
    ).astype(int)
    return summary


def to_markdown(summary, snr_threshold):
    lines = [
        f"Screening rule: retain candidate when median temporal SNR >= {snr_threshold:.1f} dB. "
        "Spectral flatness is descriptive only.",
        "",
        "| Dataset | Recordings | Median SNR (dB) | SNR IQR | Median spectral flatness | Flatness IQR | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n_recordings'])} | "
            f"{row['median_snr_db']:.2f} | {row['snr_q1']:.2f}-{row['snr_q3']:.2f} | "
            f"{row['median_spectral_flatness']:.4f} | "
            f"{row['flatness_q1']:.4f}-{row['flatness_q3']:.4f} | {row['decision']} |"
        )

    return "\n".join(lines)


def selftest():
    require_packages()
    rng = np.random.default_rng(0)
    sr = DEFAULT_SR
    t = np.arange(sr * 4) / sr

    clean = rng.normal(0, 0.002, len(t))
    for start in (0.5, 1.5, 2.5):
        mask = (t >= start) & (t < start + 0.25)
        clean[mask] += 0.8 * np.sin(2 * np.pi * 5000 * t[mask])

    noisy = rng.normal(0, 0.25, len(t))

    _, _, clean_snr = calculate_temporal_snr(clean)
    _, _, noisy_snr = calculate_temporal_snr(noisy)
    clean_flat, _ = calculate_flatness(clean)
    noisy_flat, _ = calculate_flatness(noisy)

    print(f"clean tone: SNR={clean_snr:.2f} dB, flatness={clean_flat:.4f}")
    print(f"noise     : SNR={noisy_snr:.2f} dB, flatness={noisy_flat:.4f}")
    assert clean_flat < noisy_flat
    print("OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="folder with one subfolder per species/dataset")
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="recordings sampled per dataset; 0 means all (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sr", type=int, default=DEFAULT_SR)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="maximum seconds loaded per recording; 0 means full file (default: 60)",
    )
    parser.add_argument(
        "--active-percentile",
        type=float,
        default=75.0,
        help="RMS percentile selecting high-energy frames for flatness (default: 75)",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=4.0,
        help="provisional median temporal-SNR threshold in dB (default: 4)",
    )
    parser.add_argument("--recording-out", default="recording_screen.csv")
    parser.add_argument("--species-out", default="species_screen.csv")
    parser.add_argument("--markdown-out", default="species_screen.md")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    require_packages()

    if args.selftest:
        selftest()
        return
    if not args.root:
        parser.error("--root is required unless --selftest is used")

    items = collect_files(args.root)
    if not items:
        sys.exit("no supported audio files found")

    sampled = sample_per_dataset(items, args.sample, args.seed)
    print(
        f"[info] found {len(items)} files across "
        f"{len(set(dataset for dataset, _ in items))} datasets"
    )
    print(f"[info] screening {len(sampled)} files after sampling")

    recordings = screen_recordings(
        sampled,
        sr=args.sr,
        max_seconds=args.max_seconds,
        active_percentile=args.active_percentile,
    )
    recordings.to_csv(args.recording_out, index=False)

    summary = summarize_species(recordings, args.snr_threshold)
    summary.to_csv(args.species_out, index=False)

    markdown = to_markdown(summary, args.snr_threshold)
    with open(args.markdown_out, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    print("\n" + markdown)
    print(f"\n[written] {args.recording_out}")
    print(f"[written] {args.species_out}")
    print(f"[written] {args.markdown_out}")
    print(
        "\nUse median temporal SNR to justify recording-quality screening. "
        "Use spectral flatness only to show whether vocalizations are relatively tonal or broadband."
    )


if __name__ == "__main__":
    main()

"""Shared config for the Contour+DTW baseline: paths, per-species spectrogram
params, the times_extended timebase-correction constants, and audio-path
resolution. Values below were read directly from each main_<X>_revised.py /
diagnosed by the orchestrating session -- see README_contour_dtw.md for the
provenance note on times_extended.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(__file__).resolve().parent

SEGMENTS_CSV = REPO_ROOT / "diagnosis" / "chapter1_diagnosis" / "segments.csv"  # FIX 1: full 165/165 coverage
ORIGINAL_SEQUENCES_CSV = (
    REPO_ROOT / "diagnosis" / "chapter1_pitch_diagnostic" / "evaluation_inputs" / "original_token_sequences.csv"
)

YNCC_RECONSTRUCTED_DIR = REPO_ROOT / "datasets_link" / "yncc_reconstructed_audio"
YNCC_STITCHED_DIR = BASELINE_ROOT / "yncc_stitched_audio"

CONTOURS_DIR = BASELINE_ROOT / "contours"
DIST_DIR = BASELINE_ROOT / "distance_matrices"
SANITY_DIR = BASELINE_ROOT / "sanity_checks"

SILENCE_TOKENS = {"ZERO", "SILENCE"}

# dataset -> (lyFreqlim_hz, uyFreqlim_hz, noise_threshold, amp_threshold), read from main_<X>_revised.py
SPECIES_PARAMS: dict[str, dict[str, float]] = {
    "1017_plots47_TEWA_1017":      dict(ly=3000,  uy=11000, noise=0.65, amp=0.25),
    "1017_plots47_WTSP_1017":      dict(ly=2000,  uy=8000,  noise=0.65, amp=0.25),
    "1017_plots47_Chickadee_1017": dict(ly=2000,  uy=8000,  noise=0.65, amp=0.25),
    "1017_plots47_YWAR_1017":      dict(ly=3000,  uy=9000,  noise=0.65, amp=0.25),
    "1017_plots47_YRWA_1017":      dict(ly=2500,  uy=7000,  noise=0.65, amp=0.25),
    "1017_plots47_Tseet_1017":     dict(ly=4000,  uy=10000, noise=0.05, amp=0.25),
    "1017_plots47_feebee_1017":    dict(ly=1000,  uy=7000,  noise=0.65, amp=0.25),
    "1017_plots47_CoCurate_1017":  dict(ly=1000,  uy=10000, noise=0.1,  amp=0.25),
    "1017_plots47_RNCurate_1017":  dict(ly=1000,  uy=10000, noise=0.1,  amp=0.25),
    "1017_plots47_SLCurate_1017":  dict(ly=1000,  uy=10000, noise=0.1,  amp=0.25),
    "1017_plots47_UCurate_1017":   dict(ly=1000,  uy=10000, noise=0.1,  amp=0.25),
}

# dataset -> times_extended (FIX 2: stored segment start/end are in this stretched
# timebase; real_seconds = stored / times_extended). Read from each main_<X>_revised.py.
TIMES_EXTENDED: dict[str, int] = {
    "1017_plots47_TEWA_1017":      20,
    "1017_plots47_WTSP_1017":      10,
    "1017_plots47_Chickadee_1017": 10,
    "1017_plots47_YWAR_1017":      10,
    "1017_plots47_YRWA_1017":      20,
    "1017_plots47_Tseet_1017":     10,
    "1017_plots47_feebee_1017":    10,
    "1017_plots47_CoCurate_1017":  10,
    "1017_plots47_RNCurate_1017":  10,
    "1017_plots47_SLCurate_1017":  10,
    "1017_plots47_UCurate_1017":   10,
}

YNCC_DATASETS = {
    "1017_plots47_CoCurate_1017": "Co",
    "1017_plots47_RNCurate_1017": "RN",
    "1017_plots47_SLCurate_1017": "SL",
    "1017_plots47_UCurate_1017":  "U",
}

_SPLIT_RE = re.compile(r"^(?P<prefix>.+)_split_\d+$")


def resolve_audio_path(dataset: str, song_id: str, file_path: str) -> tuple[Path | None, str]:
    """Return (path_or_None, note). note explains resolution method / failure reason."""
    if dataset not in YNCC_DATASETS:
        p = REPO_ROOT / file_path
        if p.exists():
            return p, "direct file_path"
        return None, f"non-YNCC file_path does not exist: {p}"

    label = YNCC_DATASETS[dataset]
    m = _SPLIT_RE.match(song_id)
    prefix = m.group("prefix") if m else song_id

    if label in ("Co", "U"):
        p = YNCC_RECONSTRUCTED_DIR / label / f"{song_id}.wav"
        if p.exists():
            return p, "datasets_link/yncc_reconstructed_audio (direct combined file)"
        return None, f"no reconstructed combined file at {p}"

    # RN / SL: stitched audio, matched by prefix (strip _split_NNN)
    p = YNCC_STITCHED_DIR / label / f"{prefix}.wav"
    if p.exists():
        return p, f"contour_dtw_baseline/yncc_stitched_audio (stitched clips, prefix={prefix})"
    return None, f"no source clips anywhere for prefix={prefix} (no stitched file at {p})"

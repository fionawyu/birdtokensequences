#!/usr/bin/env python3
"""Run pitch-height diagnostics without changing the coarse vocabulary.

The original vocabulary encodes coarse contour shape. This script adds average
frequency as an acoustic diagnostic attribute and creates token_pitch only for
an exploratory comparison against the unchanged original tokens.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vocabulary_sufficiency_diagnosis import (  # noqa: E402
    build_summary,
    compute_global_token_dist_metrics,
    compute_sequence_entropy_metrics,
    compute_token_dist_metrics,
    load_valid_segments,
)


SEQUENCE_COLUMNS = [
    "dataset",
    "song_id",
    "label",
    "region",
    "mode",
    "file_path",
    "token_sequence",
]
SILENCE_TOKENS = {"ZERO", "SILENCE"}
PITCH_BINS = ("LOW", "MID", "HIGH")
MAT_VARIABLE_RE = re.compile(r"^(?P<root>\w+)(?:\[(?P<index>\d+)\])?(?:\.(?P<field>\w+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add pitch diagnostics to Chapter 1 vocabulary analysis."
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=Path("diagnosis/chapter1_diagnosis_matfeat/segments.csv"),
    )
    parser.add_argument(
        "--sequences-csv",
        type=Path,
        default=Path("diagnosis/chapter1_diagnosis_matfeat/sequences.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("diagnosis/chapter1_pitch_diagnostic"),
    )
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--n-shuffle", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--feature-prefix", default="feature_")
    return parser.parse_args()


def load_mat_cached(mat_path: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(mat_path, str) or not mat_path:
        return None
    if mat_path not in cache:
        try:
            cache[mat_path] = sio.loadmat(
                mat_path, squeeze_me=True, struct_as_record=False
            )
        except Exception as exc:
            print(f"WARNING: could not load MAT file {mat_path}: {exc}", file=sys.stderr)
            cache[mat_path] = {}
    return cache[mat_path] or None


def parse_mat_variable(variable: object) -> tuple[str, int | None, str | None] | None:
    if not isinstance(variable, str) or not variable:
        return None
    match = MAT_VARIABLE_RE.match(variable)
    if match is None:
        return None
    index = match.group("index")
    return (
        match.group("root"),
        int(index) if index is not None else None,
        match.group("field"),
    )


def get_mat_object(mat_data: dict[str, Any], mat_variable: object) -> tuple[np.ndarray, np.ndarray | None] | None:
    parsed = parse_mat_variable(mat_variable)
    if parsed is None:
        return None
    root, index, field = parsed
    if root not in mat_data:
        return None

    value = mat_data[root]
    if index is not None:
        items = np.ravel(value)
        if index >= len(items):
            return None
        item = items[index]
        if field is None or not hasattr(item, field):
            return None
        spec = getattr(item, field)
        freqs = getattr(item, "freqs", None)
    else:
        spec = getattr(value, field) if field and hasattr(value, field) else value
        freqs = getattr(value, "freqs", None) if hasattr(value, "freqs") else None

    if not isinstance(spec, np.ndarray) or spec.ndim != 2:
        return None
    if isinstance(freqs, np.ndarray):
        freqs = np.asarray(freqs, dtype=float).reshape(-1)
    else:
        freqs = None
    return np.asarray(spec, dtype=float), freqs


def segment_average_frequency(row: pd.Series, mat_cache: dict[str, dict[str, Any]]) -> float:
    mat_data = load_mat_cached(str(row.get("mat_path", "")), mat_cache)
    if mat_data is None:
        return float("nan")

    selected = get_mat_object(mat_data, row.get("mat_variable"))
    if selected is None:
        return float("nan")
    spec, freqs = selected
    if freqs is None or len(freqs) != spec.shape[0]:
        return float("nan")

    try:
        x0 = int(row["crop_x0"])
        x1 = int(row["crop_x1"])
    except (TypeError, ValueError):
        return float("nan")

    n_time = spec.shape[1]
    x0 = max(0, min(n_time, x0))
    x1 = max(0, min(n_time, x1))
    if x1 <= x0:
        return float("nan")

    patch = np.asarray(spec[:, x0:x1], dtype=float)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.where(patch > 0.0, patch, 0.0)

    valid_freq_mask = np.isfinite(freqs) & (freqs > 0.0)
    if not valid_freq_mask.any():
        return float("nan")

    patch = patch[valid_freq_mask, :]
    valid_freqs = freqs[valid_freq_mask]
    frame_energy = patch.sum(axis=0)
    valid_frame_mask = frame_energy > 0.0
    if not valid_frame_mask.any():
        return float("nan")

    weighted_sum = (valid_freqs[:, None] * patch).sum(axis=0)
    frame_avg = weighted_sum[valid_frame_mask] / frame_energy[valid_frame_mask]
    frame_avg = frame_avg[np.isfinite(frame_avg) & (frame_avg > 0.0)]
    if len(frame_avg) == 0:
        return float("nan")
    return float(frame_avg.mean())


def add_avg_frequency(segments: pd.DataFrame) -> pd.DataFrame:
    mat_cache: dict[str, dict[str, Any]] = {}
    enriched = segments.copy()
    enriched["avg_freq_hz"] = [
        segment_average_frequency(row, mat_cache) for _, row in enriched.iterrows()
    ]
    return enriched


def assign_pitch_bins(segments: pd.DataFrame) -> pd.DataFrame:
    enriched = segments.copy()
    enriched["pitch_bin"] = pd.NA

    contour_mask = (
        ~enriched["token"].astype(str).str.upper().isin(SILENCE_TOKENS)
        & enriched["avg_freq_hz"].notna()
        & np.isfinite(enriched["avg_freq_hz"])
    )

    for dataset, dataset_rows in enriched.loc[contour_mask].groupby("dataset"):
        values = dataset_rows["avg_freq_hz"].astype(float)
        if values.empty:
            continue
        q_low = float(values.quantile(1.0 / 3.0))
        q_high = float(values.quantile(2.0 / 3.0))

        def bin_value(value: float) -> str:
            if value <= q_low:
                return "LOW"
            if value <= q_high:
                return "MID"
            return "HIGH"

        enriched.loc[dataset_rows.index, "pitch_bin"] = values.map(bin_value)

    return enriched


def make_token_pitch(row: pd.Series) -> str:
    token = str(row["token"])
    upper = token.upper()
    if upper in SILENCE_TOKENS:
        return token
    pitch_bin = row.get("pitch_bin")
    if pd.isna(pitch_bin):
        return token
    return f"{token}_{pitch_bin}"


def add_token_pitch(segments: pd.DataFrame) -> pd.DataFrame:
    enriched = segments.copy()
    enriched["token_pitch"] = enriched.apply(make_token_pitch, axis=1)
    return enriched


def summarize_pitch_by_token(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for token, group in segments.groupby("token", dropna=False):
        freq = pd.to_numeric(group["avg_freq_hz"], errors="coerce")
        bin_counts = group["pitch_bin"].value_counts(dropna=True).to_dict()
        binned_total = sum(int(bin_counts.get(bin_name, 0)) for bin_name in PITCH_BINS)
        strong_bins = [
            bin_name
            for bin_name in PITCH_BINS
            if binned_total > 0 and int(bin_counts.get(bin_name, 0)) / binned_total >= 0.20
        ]
        rows.append(
            {
                "token": token,
                "n_segments": len(group),
                "valid_avg_freq_segments": int(freq.notna().sum()),
                "mean_avg_freq_hz": freq.mean(),
                "std_avg_freq_hz": freq.std(),
                "min_avg_freq_hz": freq.min(),
                "max_avg_freq_hz": freq.max(),
                "n_LOW": int(bin_counts.get("LOW", 0)),
                "n_MID": int(bin_counts.get("MID", 0)),
                "n_HIGH": int(bin_counts.get("HIGH", 0)),
                "pitch_mixed": len(strong_bins) >= 2,
            }
        )
    return pd.DataFrame(rows).sort_values("token")


def build_token_pitch_sequences(segments: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["dataset", "song_id", "segment_id"]
    sorted_segments = segments.sort_values(sort_cols)
    rows = []
    group_cols = ["dataset", "song_id", "label", "region", "mode", "file_path"]
    for keys, group in sorted_segments.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys, strict=False))
        row["token_sequence"] = " ".join(group["token_pitch"].astype(str))
        rows.append(row)
    return pd.DataFrame(rows, columns=SEQUENCE_COLUMNS)


def run_diagnosis(
    name: str,
    segments: pd.DataFrame,
    sequences: pd.DataFrame,
    out_dir: Path,
    min_count: int,
    n_shuffle: int,
    random_seed: int,
    feature_prefix: str,
) -> pd.DataFrame:
    input_dir = out_dir / "evaluation_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    segments_path = input_dir / f"{name}_segments.csv"
    sequences_path = input_dir / f"{name}_sequences.csv"
    segments.to_csv(segments_path, index=False)
    sequences.to_csv(sequences_path, index=False)

    valid_segments, feature_cols = load_valid_segments(segments_path, feature_prefix)
    token_details, dist = compute_token_dist_metrics(valid_segments, feature_cols, min_count)
    global_token_dist = compute_global_token_dist_metrics(valid_segments, feature_cols, min_count)
    entropy, transitions = compute_sequence_entropy_metrics(
        sequences, n_shuffle, random_seed
    )
    summary = build_summary(dist, entropy)

    summary.to_csv(out_dir / f"{name}_diagnosis_summary.csv", index=False)
    token_details.to_csv(out_dir / f"{name}_token_dist_details.csv", index=False)
    global_token_dist.to_csv(out_dir / f"{name}_global_token_dist_summary.csv", index=False)
    transitions.to_csv(out_dir / f"{name}_transition_counts.csv", index=False)
    return summary


def compare_summaries(original: pd.DataFrame, token_pitch: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        original,
        token_pitch,
        on="dataset",
        how="outer",
        suffixes=("_original", "_token_pitch"),
    )
    for metric in ["Dist", "H_obs", "H_shuffle_mean", "entropy_gap"]:
        left = f"{metric}_original"
        right = f"{metric}_token_pitch"
        if left in merged.columns and right in merged.columns:
            merged[f"{metric}_delta_token_pitch_minus_original"] = (
                merged[right] - merged[left]
            )
    return merged.sort_values("dataset")


def write_readme(out_dir: Path) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        """# Chapter 1 Pitch Diagnostic

This folder is a diagnostic extension of the Chapter 1 vocabulary analysis.

- The original vocabulary encodes coarse contour shape only.
- Average frequency is added as an acoustic diagnostic attribute (`avg_freq_hz`).
- `pitch_bin` splits non-silence segments into dataset-level LOW/MID/HIGH bins.
- `token_pitch` is not the main vocabulary and does not replace `token`.
- The original edit-distance pipeline and sequence-level evaluation remain unchanged.
- If `token_pitch` reduces within-token heterogeneity or improves separation, it suggests
  that pitch height should be considered in a refined vocabulary.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = pd.read_csv(args.segments_csv)
    sequences = pd.read_csv(args.sequences_csv)

    enriched = add_avg_frequency(segments)
    enriched = assign_pitch_bins(enriched)
    enriched = add_token_pitch(enriched)
    enriched.to_csv(out_dir / "segment_features_with_pitch.csv", index=False)

    summary = summarize_pitch_by_token(enriched)
    summary.to_csv(out_dir / "token_pitch_summary.csv", index=False)

    original_summary = run_diagnosis(
        name="original_token",
        segments=enriched,
        sequences=sequences,
        out_dir=out_dir,
        min_count=args.min_count,
        n_shuffle=args.n_shuffle,
        random_seed=args.random_seed,
        feature_prefix=args.feature_prefix,
    )

    token_pitch_segments = enriched.copy()
    token_pitch_segments["token"] = token_pitch_segments["token_pitch"]
    token_pitch_sequences = build_token_pitch_sequences(enriched)
    token_pitch_summary = run_diagnosis(
        name="token_pitch",
        segments=token_pitch_segments,
        sequences=token_pitch_sequences,
        out_dir=out_dir,
        min_count=args.min_count,
        n_shuffle=args.n_shuffle,
        random_seed=args.random_seed,
        feature_prefix=args.feature_prefix,
    )

    comparison = compare_summaries(original_summary, token_pitch_summary)
    comparison.to_csv(out_dir / "original_vs_pitch_token_comparison.csv", index=False)
    write_readme(out_dir)

    print(f"Wrote {out_dir / 'segment_features_with_pitch.csv'}")
    print(f"Wrote {out_dir / 'token_pitch_summary.csv'}")
    print(f"Wrote {out_dir / 'original_vs_pitch_token_comparison.csv'}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()

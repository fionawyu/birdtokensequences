#!/usr/bin/env python3
"""Compute vocabulary sufficiency diagnostics for tokenized bird vocalizations.

Outputs:
  - diagnosis_summary.csv
  - token_dist_details.csv
  - transition_counts.csv, when requested
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEGMENT_METADATA_COLUMNS = {"dataset", "song_id", "segment_id", "token", "label"}
REQUIRED_SEGMENT_COLUMNS = {"dataset", "song_id", "segment_id", "token"}
REQUIRED_SEQUENCE_COLUMNS = {"dataset", "song_id", "label", "token_sequence"}
TOKEN_DISPLAY_NAMES = {"ZERO": "SILENCE"}


def warn(message: str) -> None:
    """Print a warning that is easy to see in CLI logs."""
    print(f"WARNING: {message}", file=sys.stderr)


def validate_columns(df: pd.DataFrame, required: set[str], csv_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {', '.join(missing)}")


def find_feature_columns(
    segments: pd.DataFrame, feature_prefix: str
) -> tuple[list[str], str]:
    prefixed = [col for col in segments.columns if col.startswith(feature_prefix)]
    if prefixed:
        return sorted(prefixed, key=feature_sort_key(feature_prefix)), "prefixed"

    numeric_cols = [
        col
        for col in segments.select_dtypes(include=[np.number]).columns
        if col not in SEGMENT_METADATA_COLUMNS
    ]
    if not numeric_cols:
        raise ValueError(
            "No feature columns found. Expected columns with prefix "
            f"{feature_prefix!r} or numeric feature columns outside metadata."
        )
    return numeric_cols, "numeric"


def feature_sort_key(feature_prefix: str):
    def key(col: str) -> tuple[int, int | str]:
        suffix = col[len(feature_prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, col)

    return key


def load_valid_segments(
    csv_path: Path, feature_prefix: str
) -> tuple[pd.DataFrame, list[str]]:
    segments = pd.read_csv(csv_path)
    validate_columns(segments, REQUIRED_SEGMENT_COLUMNS, str(csv_path))

    feature_cols, source = find_feature_columns(segments, feature_prefix)
    if source == "numeric":
        warn(
            "No feature-prefix columns found; using numeric columns as features: "
            + ", ".join(feature_cols)
        )

    working = segments.copy()
    working["token"] = working["token"].astype("string")
    working["token"] = working["token"].str.strip()
    working.loc[working["token"].eq(""), "token"] = pd.NA

    before = len(working)
    working = working.dropna(subset=["dataset", "token", *feature_cols])
    dropped_missing = before - len(working)
    if dropped_missing:
        warn(f"Dropped {dropped_missing} segment rows with missing token or features.")

    for col in feature_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    before = len(working)
    feature_values = working[feature_cols].to_numpy(dtype=float)
    finite_mask = np.isfinite(feature_values).all(axis=1)
    working = working.loc[finite_mask].copy()
    dropped_nonfinite = before - len(working)
    if dropped_nonfinite:
        warn(f"Dropped {dropped_nonfinite} segment rows with non-finite feature values.")

    before = len(working)
    norms = np.linalg.norm(working[feature_cols].to_numpy(dtype=float), axis=1)
    nonzero_mask = norms > 0.0
    working = working.loc[nonzero_mask].copy()
    dropped_zero = before - len(working)
    if dropped_zero:
        warn(f"Dropped {dropped_zero} segment rows with zero feature-vector norm.")

    return working, feature_cols


def cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / norms
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def compute_token_dist_metrics(
    segments: pd.DataFrame, feature_cols: list[str], min_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details: list[dict[str, object]] = []
    segment_counts = segments.groupby("dataset", dropna=False).size().to_dict()

    for (dataset, token), group in segments.groupby(["dataset", "token"], dropna=False):
        token_count = len(group)
        if token_count < min_count:
            continue

        features = group[feature_cols].to_numpy(dtype=float)
        distances = cosine_distance_matrix(features)
        medoid_idx = int(np.argmin(distances.sum(axis=1)))
        d_j = float(distances[:, medoid_idx].mean())
        details.append(
            {
                "dataset": dataset,
                "token": token,
                "token_count": token_count,
                "D_j": d_j,
            }
        )

    details_df = pd.DataFrame(
        details, columns=["dataset", "token", "token_count", "D_j"]
    )

    datasets_df = pd.DataFrame(
        [
            {"dataset": dataset, "n_segments": n_segments}
            for dataset, n_segments in segment_counts.items()
        ]
    )
    if datasets_df.empty:
        dist_df = pd.DataFrame(columns=["dataset", "Dist", "n_segments", "n_tokens_used"])
    elif details_df.empty:
        dist_df = datasets_df.copy()
        dist_df["Dist"] = np.nan
        dist_df["n_tokens_used"] = 0
    else:
        token_summary = (
            details_df.groupby("dataset", as_index=False)
            .agg(Dist=("D_j", "mean"), n_tokens_used=("token", "count"))
        )
        dist_df = pd.merge(datasets_df, token_summary, on="dataset", how="left")
        dist_df["n_tokens_used"] = dist_df["n_tokens_used"].fillna(0).astype(int)
        dist_df = dist_df.sort_values("dataset")

    for dataset, n_segments in sorted(segment_counts.items()):
        if n_segments < min_count:
            warn(
                f"Dataset {dataset!r} has only {n_segments} valid segments; "
                "Dist may be unavailable."
            )

    datasets_with_tokens = set(details_df["dataset"]) if not details_df.empty else set()
    for dataset in sorted(segment_counts):
        if dataset not in datasets_with_tokens:
            warn(
                f"Dataset {dataset!r} has no tokens with at least min_count={min_count} "
                "valid segments; Dist is NaN."
            )

    return details_df, dist_df


def compute_global_token_dist_metrics(
    segments: pd.DataFrame, feature_cols: list[str], min_count: int
) -> pd.DataFrame:
    """Compute token distances across all datasets, without species grouping."""
    rows: list[dict[str, object]] = []

    for token, group in segments.groupby("token", dropna=False):
        token_count = len(group)
        if token_count < min_count:
            continue

        features = group[feature_cols].to_numpy(dtype=float)
        distances = cosine_distance_matrix(features)
        medoid_idx = int(np.argmin(distances.sum(axis=1)))
        d_j = float(distances[:, medoid_idx].mean())

        rows.append(
            {
                "token": token,
                "display_token": TOKEN_DISPLAY_NAMES.get(str(token).upper(), token),
                "token_count": token_count,
                "D_j": d_j,
            }
        )

    return pd.DataFrame(
        rows, columns=["token", "display_token", "token_count", "D_j"]
    ).sort_values("token")


def parse_token_sequence(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return str(value).split()


def transition_counts_from_sequences(
    sequences: Iterable[list[str]],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for tokens in sequences:
        counts.update(zip(tokens[:-1], tokens[1:]))
    return counts


def entropy_rate_from_transition_counts(counts: Counter[tuple[str, str]]) -> float:
    total_bigrams = sum(counts.values())
    if total_bigrams == 0:
        return float("nan")

    outgoing: defaultdict[str, int] = defaultdict(int)
    for (prev_token, _), count in counts.items():
        outgoing[prev_token] += count

    entropy = 0.0
    for (prev_token, _), count in counts.items():
        p_joint = count / total_bigrams
        p_next_given_prev = count / outgoing[prev_token]
        entropy -= p_joint * np.log2(p_next_given_prev)
    return float(entropy)


def compute_sequence_entropy_metrics(
    sequences_df: pd.DataFrame, n_shuffle: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(sequences_df, REQUIRED_SEQUENCE_COLUMNS, "sequences CSV")

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []

    for dataset, group in sequences_df.groupby("dataset", dropna=False):
        parsed_sequences = [parse_token_sequence(value) for value in group["token_sequence"]]
        valid_sequences = [tokens for tokens in parsed_sequences if len(tokens) >= 2]
        n_sequences = len(valid_sequences)

        if n_sequences == 0:
            warn(
                f"Dataset {dataset!r} has no valid sequences with at least 2 tokens; "
                "entropy metrics are NaN."
            )
            rows.append(
                {
                    "dataset": dataset,
                    "H_obs": np.nan,
                    "H_shuffle_mean": np.nan,
                    "H_shuffle_std": np.nan,
                    "entropy_gap": np.nan,
                    "n_sequences": 0,
                }
            )
            continue

        if n_sequences < 2:
            warn(
                f"Dataset {dataset!r} has only {n_sequences} valid sequence; "
                "entropy estimates may be unstable."
            )

        obs_counts = transition_counts_from_sequences(valid_sequences)
        h_obs = entropy_rate_from_transition_counts(obs_counts)

        for (prev_token, next_token), count in sorted(obs_counts.items()):
            transition_rows.append(
                {
                    "dataset": dataset,
                    "prev_token": prev_token,
                    "next_token": next_token,
                    "count": count,
                }
            )

        shuffled_entropies: list[float] = []
        for _ in range(n_shuffle):
            shuffled_sequences = []
            for tokens in valid_sequences:
                shuffled = np.array(tokens, dtype=object).copy()
                rng.shuffle(shuffled)
                shuffled_sequences.append(shuffled.tolist())
            shuffled_counts = transition_counts_from_sequences(shuffled_sequences)
            shuffled_entropies.append(entropy_rate_from_transition_counts(shuffled_counts))

        if shuffled_entropies:
            h_shuffle_mean = float(np.mean(shuffled_entropies))
            h_shuffle_std = float(np.std(shuffled_entropies))
        else:
            h_shuffle_mean = np.nan
            h_shuffle_std = np.nan

        rows.append(
            {
                "dataset": dataset,
                "H_obs": h_obs,
                "H_shuffle_mean": h_shuffle_mean,
                "H_shuffle_std": h_shuffle_std,
                "entropy_gap": h_shuffle_mean - h_obs,
                "n_sequences": n_sequences,
            }
        )

    entropy_df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "H_obs",
            "H_shuffle_mean",
            "H_shuffle_std",
            "entropy_gap",
            "n_sequences",
        ],
    )
    transition_df = pd.DataFrame(
        transition_rows, columns=["dataset", "prev_token", "next_token", "count"]
    )
    return entropy_df, transition_df


def build_summary(dist_df: pd.DataFrame, entropy_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.merge(entropy_df, dist_df, on="dataset", how="outer")
    summary["n_sequences"] = summary["n_sequences"].fillna(0).astype(int)
    summary["n_segments"] = summary["n_segments"].fillna(0).astype(int)
    summary["n_tokens_used"] = summary["n_tokens_used"].fillna(0).astype(int)

    return summary[
        [
            "dataset",
            "Dist",
            "H_obs",
            "H_shuffle_mean",
            "H_shuffle_std",
            "entropy_gap",
            "n_sequences",
            "n_segments",
            "n_tokens_used",
        ]
    ].sort_values("dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute token acoustic consistency and sequence entropy diagnostics "
            "for bird vocalization token sequences."
        )
    )
    parser.add_argument("--segments-csv", required=True, type=Path)
    parser.add_argument("--sequences-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-count", default=2, type=int)
    parser.add_argument("--n-shuffle", default=200, type=int)
    parser.add_argument("--random-seed", default=0, type=int)
    parser.add_argument("--feature-prefix", default="feature_")
    parser.add_argument(
        "--write-transition-counts",
        action="store_true",
        help="Also write transition_counts.csv for observed bigrams.",
    )
    return parser.parse_args()


def validate_input_path(path: Path, arg_name: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {arg_name} does not exist: {path}. "
            f"Current working directory is {Path.cwd()}. "
            "Use an absolute path or a path relative to the directory where "
            "you run this command."
        )
    if not path.is_file():
        raise SystemExit(f"ERROR: {arg_name} is not a file: {path}")


def main() -> None:
    args = parse_args()
    if args.min_count < 1:
        raise ValueError("--min-count must be at least 1")
    if args.n_shuffle < 0:
        raise ValueError("--n-shuffle must be non-negative")
    validate_input_path(args.segments_csv, "--segments-csv")
    validate_input_path(args.sequences_csv, "--sequences-csv")

    segments, feature_cols = load_valid_segments(args.segments_csv, args.feature_prefix)
    sequences = pd.read_csv(args.sequences_csv)

    token_details_df, dist_df = compute_token_dist_metrics(
        segments, feature_cols, args.min_count
    )
    global_token_dist_df = compute_global_token_dist_metrics(
        segments, feature_cols, args.min_count
    )
    entropy_df, transition_df = compute_sequence_entropy_metrics(
        sequences, args.n_shuffle, args.random_seed
    )
    summary_df = build_summary(dist_df, entropy_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "diagnosis_summary.csv"
    token_details_path = args.out_dir / "token_dist_details.csv"
    global_token_dist_path = args.out_dir / "global_token_dist_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    token_details_df.to_csv(token_details_path, index=False)
    global_token_dist_df.to_csv(global_token_dist_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {token_details_path}")
    print(f"Wrote {global_token_dist_path}")

    if args.write_transition_counts:
        transition_path = args.out_dir / "transition_counts.csv"
        transition_df.to_csv(transition_path, index=False)
        print(f"Wrote {transition_path}")


if __name__ == "__main__":
    main()

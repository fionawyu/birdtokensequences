"""Step A — rebuild the attribute model with the six-feature set + k-medoids.

Reuses the existing formula evaluator (funcWrappers.wrapExtParseEvalBatch,
via the same wrapper scripts/run_committee_formula_baseline.py already uses)
so this is not a new extraction pipeline, per the task spec. Nothing here
alters segments.csv or any existing script's output.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from combined_model.common import (
    ATTRIBUTE_BASELINE_SEEDS,
    DATASETS,
    REPO_ROOT,
    SEGMENTS_CSV,
    SELECTED_SEEDS,
    SILENCE_TOKENS,
    clustering_accuracy,
    kmedoids_fit,
)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FORMULA_SAMPLE_POINTS = 64

NEW_FEATURES = ["min_freq", "max_freq", "peak_freq", "bandwidth", "duration", "freq_modulation"]


def evaluate_formula(formula: str, n_points: int = FORMULA_SAMPLE_POINTS) -> np.ndarray:
    """Sample f-hat(t) on the same normalized-time frame grid used elsewhere
    in this repo for formula evaluation (identical convention to
    scripts/run_committee_formula_baseline.py:evaluate_formula, since the
    formula is fitted on segment-normalized time)."""
    if not formula or str(formula).strip() in {"", "nan"}:
        return np.full(n_points, np.nan)
    if str(formula).strip() == "0":
        return np.zeros(n_points, dtype=float)

    from funcWrappers import wrapExtParseEvalBatch

    t_values = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    try:
        values = np.asarray(wrapExtParseEvalBatch(str(formula), t_values), dtype=float)
    except Exception:
        return np.full(n_points, np.nan)
    return np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)


def freq_modulation(values: np.ndarray, duration: float) -> float:
    """Number of sign changes in the first difference of f-hat(t), divided
    by segment duration in seconds."""
    finite = values[np.isfinite(values)]
    if finite.size < 3 or not np.isfinite(duration) or duration <= 1e-9:
        return np.nan
    diffs = np.diff(finite)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    n_changes = int(np.sum(signs[:-1] != signs[1:]))
    return n_changes / duration


def compute_segment_features(segments: pd.DataFrame) -> pd.DataFrame:
    """Step A1: the six per-unit features from f-hat(t) and segment bounds."""
    rows = []
    for _, row in segments.iterrows():
        token = str(row.get("token", ""))
        is_silence = token.upper() in SILENCE_TOKENS
        duration = float(row.get("duration", np.nan))

        if is_silence:
            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "song_id": row.get("song_id"),
                    "segment_id": row.get("segment_id"),
                    "token": token,
                    "is_silence": True,
                    "min_freq": np.nan,
                    "max_freq": np.nan,
                    "peak_freq": np.nan,
                    "bandwidth": np.nan,
                    "duration": duration,
                    "freq_modulation": np.nan,
                }
            )
            continue

        values = evaluate_formula(str(row.get("subformula", "")))
        finite = values[np.isfinite(values)]
        min_freq = float(np.min(finite)) if finite.size else np.nan
        max_freq = float(np.max(finite)) if finite.size else np.nan
        peak_freq = float(np.mean(finite)) if finite.size else np.nan
        bandwidth = max_freq - min_freq if finite.size else np.nan

        rows.append(
            {
                "dataset": row.get("dataset"),
                "song_id": row.get("song_id"),
                "segment_id": row.get("segment_id"),
                "token": token,
                "is_silence": False,
                "min_freq": min_freq,
                "max_freq": max_freq,
                "peak_freq": peak_freq,
                "bandwidth": bandwidth,
                "duration": duration,
                "freq_modulation": freq_modulation(values, duration),
            }
        )
    return pd.DataFrame(rows)


def aggregate_recording_features(segment_features: pd.DataFrame) -> pd.DataFrame:
    """Step A2: mean/sd/min/max of the six features + n_units + n_silences
    = 26 dimensions per recording."""
    rows = []
    for (dataset, song_id), group in segment_features.groupby(["dataset", "song_id"]):
        units = group[~group["is_silence"]]
        silences = group[group["is_silence"]]
        row: dict[str, object] = {
            "dataset": dataset,
            "song_id": song_id,
            "n_units": len(units),
            "n_silences": len(silences),
        }
        for feature in NEW_FEATURES:
            values = pd.to_numeric(units[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{feature}_sd"] = float(values.std(ddof=0)) if len(values) else np.nan
            row[f"{feature}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{feature}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def attribute_distance_matrix(recording_features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Step A3: z-score within the setting being evaluated, then Euclidean."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(recording_features[feature_cols].to_numpy(dtype=float))
    matrix = scaler.fit_transform(matrix)
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def evaluate_attribute_model(
    recording_features: pd.DataFrame,
    feature_cols: list[str],
    seeds_by_setting: dict[str, list[int]],
    method_name: str,
) -> pd.DataFrame:
    rows = []
    for setting, datasets in DATASETS.items():
        subset = recording_features[recording_features["dataset"].isin(datasets)].reset_index(drop=True)
        labels = sorted(subset["dataset"].unique())
        if len(labels) < 2 or subset.empty:
            continue
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        y_true = subset["dataset"].map(label_to_int).to_numpy(dtype=int)

        dist = attribute_distance_matrix(subset, feature_cols)
        for seed in seeds_by_setting[setting]:
            y_pred = kmedoids_fit(dist, n_clusters=len(labels), seed=seed)
            acc = clustering_accuracy(y_true, y_pred) * 100.0
            rows.append(
                {
                    "setting": setting,
                    "method": method_name,
                    "seed": seed,
                    "accuracy": acc,
                    "n_recordings": len(subset),
                    "n_classes": len(labels),
                }
            )
    return pd.DataFrame(rows)


def build_new_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (segment_features, recording_features) -- the per-unit and
    per-recording feature tables from Step A1/A2, for inspection."""
    segments = pd.read_csv(SEGMENTS_CSV)
    segment_features = compute_segment_features(segments)
    recording_features = aggregate_recording_features(segment_features)
    return segment_features, recording_features


def build_new_feature_recording_table() -> pd.DataFrame:
    _, recording_features = build_new_feature_tables()
    return recording_features


def new_feature_columns() -> list[str]:
    cols = []
    for feature in NEW_FEATURES:
        cols += [f"{feature}_mean", f"{feature}_sd", f"{feature}_min", f"{feature}_max"]
    cols += ["n_units", "n_silences"]
    return cols


# ---------------------------------------------------------------------------
# Step A4 acceptance check: old (published) feature set through k-medoids
# instead of KMeans, using the genuinely-fixed baseline seeds, to isolate
# the clusterer-switch effect from the feature-set change.
# ---------------------------------------------------------------------------


def build_old_feature_recording_table() -> tuple[pd.DataFrame, list[str]]:
    from run_committee_formula_baseline import (
        aggregate_recording_features as old_aggregate,
    )
    from run_committee_formula_baseline import (
        segment_formula_features as old_segment_features,
    )

    segments = pd.read_csv(SEGMENTS_CSV)
    seg_feats = old_segment_features(segments)
    rec_feats = old_aggregate(seg_feats, n_formula_clusters=8)
    feature_cols = [
        c
        for c in rec_feats.select_dtypes(include=[np.number]).columns
        if c not in {"seed"}
    ]
    return rec_feats, feature_cols


def run_step_a() -> dict[str, pd.DataFrame]:
    new_recordings = build_new_feature_recording_table()
    new_cols = new_feature_columns()
    new_eval_selected = evaluate_attribute_model(
        new_recordings, new_cols, SELECTED_SEEDS, "AttributeNew+KMedoids(selected-seeds)"
    )
    new_eval_baseline = evaluate_attribute_model(
        new_recordings,
        new_cols,
        {s: ATTRIBUTE_BASELINE_SEEDS for s in DATASETS},
        "AttributeNew+KMedoids(baseline-seeds)",
    )

    old_recordings, old_cols = build_old_feature_recording_table()
    old_eval_baseline = evaluate_attribute_model(
        old_recordings,
        old_cols,
        {s: ATTRIBUTE_BASELINE_SEEDS for s in DATASETS},
        "AttributeOld+KMedoids(baseline-seeds)",
    )

    return {
        "new_recordings": new_recordings,
        "new_eval_selected": new_eval_selected,
        "new_eval_baseline": new_eval_baseline,
        "old_recordings": old_recordings,
        "old_eval_baseline": old_eval_baseline,
    }


if __name__ == "__main__":
    results = run_step_a()
    for name, df in results.items():
        if "eval" in name:
            print(f"== {name} ==")
            print(df.groupby("setting")["accuracy"].agg(["mean", "std", "max"]))

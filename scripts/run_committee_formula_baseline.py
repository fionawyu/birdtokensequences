#!/usr/bin/env python3
"""Formula-feature baseline for Chapter 1 committee-response analyses.

This script adds a direct formula-derived baseline without changing the
existing symbolic-token pipeline. It reads the existing diagnosis inputs,
extracts compact numerical features from each fitted segment/formula, aggregates
them to recording-level features, and evaluates those features with KMeans on
the same Chapter 1 contrast definitions used by the token clustering scripts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTS = REPO_ROOT / "diagnosis" / "chapter1_diagnosis_matfeat" / "segments.csv"
DEFAULT_SEQUENCES = REPO_ROOT / "diagnosis" / "chapter1_diagnosis_matfeat" / "sequences.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results_committee_response"
SEEDS = [42, 156, 12, 1]
MATRIX_SIDE = 32
FORMULA_SAMPLE_POINTS = 64

DATASETS = {
    "SegMix": [
        "1017_plots47_WTSP_1017",
        "1017_plots47_feebee_1017",
    ],
    "NoteMix": [
        "1017_plots47_YWAR_1017",
        "1017_plots47_YRWA_1017",
    ],
    "BCCH-var": [
        "1017_plots47_Chickadee_1017",
        "1017_plots47_Tseet_1017",
        "1017_plots47_feebee_1017",
    ],
    "Species": [
        "1017_plots47_TEWA_1017",
        "1017_plots47_WTSP_1017",
        "1017_plots47_Chickadee_1017",
        "1017_plots47_YWAR_1017",
    ],
    "AllVar": [
        "1017_plots47_Chickadee_1017",
        "1017_plots47_Tseet_1017",
        "1017_plots47_feebee_1017",
        "1017_plots47_TEWA_1017",
        "1017_plots47_WTSP_1017",
        "1017_plots47_YWAR_1017",
    ],
    "YNCCDialect": [
        "1017_plots47_CoCurate_1017",
        "1017_plots47_UCurate_1017",
        "1017_plots47_RNCurate_1017",
        "1017_plots47_SLCurate_1017",
    ],
}

OPERATORS = [
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "sqr",
    "abs",
    "neg",
    "+",
    "-",
    "*",
    "/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct formula-feature baseline.")
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--sequences-csv", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-formula-clusters", type=int, default=8)
    parser.add_argument(
        "--eval-n-init",
        type=int,
        default=1,
        help=(
            "Number of KMeans initializations for each reported evaluation seed. "
            "Use 1 so each seed corresponds to one independent clustering run."
        ),
    )
    return parser.parse_args()


def add_repo_to_path() -> None:
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def evaluate_formula(formula: str, n_points: int = FORMULA_SAMPLE_POINTS) -> np.ndarray:
    """Evaluate a formula on normalized time; return NaNs if evaluation fails."""
    if not formula or str(formula).strip() in {"", "nan"}:
        return np.full(n_points, np.nan)
    if str(formula).strip() == "0":
        return np.zeros(n_points, dtype=float)

    add_repo_to_path()
    from funcWrappers import wrapExtParseEvalBatch

    t_values = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    try:
        values = np.asarray(wrapExtParseEvalBatch(str(formula), t_values), dtype=float)
    except Exception:
        return np.full(n_points, np.nan)
    values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return values


def safe_stat(values: np.ndarray, func, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(func(finite))


def linear_rmse(values: np.ndarray) -> float:
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[finite_mask]
    y = values[finite_mask]
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def curvature(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.nan
    return float(np.mean(np.abs(np.diff(finite, n=2))))


def formula_structure_features(formula: str) -> dict[str, float]:
    text = "" if pd.isna(formula) else str(formula)
    numbers = re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", text)
    tokens = re.findall(r"[A-Za-z_+\-*/><=&|]+|-?\d+(?:\.\d+)?", text)
    rows: dict[str, float] = {
        "formula_string_length": float(len(text)),
        "formula_node_count": float(len(tokens)),
        "formula_numeric_constant_count": float(len(numbers)),
        "formula_unique_operator_count": float(sum(op in tokens for op in OPERATORS)),
    }
    for op in OPERATORS:
        safe_name = (
            op.replace("+", "plus")
            .replace("-", "minus")
            .replace("*", "mul")
            .replace("/", "div")
        )
        rows[f"op_{safe_name}_count"] = float(tokens.count(op))
    return rows


def matrix_contour_features(row: pd.Series, feature_cols: list[str]) -> dict[str, float]:
    values = row[feature_cols].to_numpy(dtype=float)
    if values.size != MATRIX_SIDE * MATRIX_SIDE or not np.isfinite(values).any():
        return {
            "matrix_start_freq": np.nan,
            "matrix_end_freq": np.nan,
            "matrix_mean_freq": np.nan,
            "matrix_bandwidth": np.nan,
            "matrix_freq_range": np.nan,
            "formula_matrix_shape_rmse": np.nan,
        }

    mat = np.nan_to_num(values.reshape(MATRIX_SIDE, MATRIX_SIDE), nan=0.0)
    weights = np.maximum(mat, 0.0)
    freq_axis = np.linspace(0.0, 1.0, MATRIX_SIDE)
    col_mass = weights.sum(axis=0)
    valid = col_mass > 1e-12
    contour = np.full(MATRIX_SIDE, np.nan)
    contour[valid] = (weights[:, valid] * freq_axis[:, None]).sum(axis=0) / col_mass[valid]
    active_freqs = np.repeat(freq_axis, (weights.sum(axis=1) > 1e-12).astype(int))

    formula_values = evaluate_formula(str(row.get("subformula", "")), MATRIX_SIDE)
    rmse = normalized_rmse(formula_values, contour)

    return {
        "matrix_start_freq": safe_stat(contour[: max(1, MATRIX_SIDE // 8)], np.nanmean),
        "matrix_end_freq": safe_stat(contour[-max(1, MATRIX_SIDE // 8) :], np.nanmean),
        "matrix_mean_freq": safe_stat(contour, np.nanmean),
        "matrix_bandwidth": float(np.mean(col_mass > 1e-12)),
        "matrix_freq_range": float(np.nanmax(contour) - np.nanmin(contour))
        if np.isfinite(contour).any()
        else np.nan,
        "formula_matrix_shape_rmse": rmse,
    }


def normalized_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    x = a[mask]
    y = b[mask]
    if np.std(x) > 1e-12:
        x = (x - np.mean(x)) / np.std(x)
    if np.std(y) > 1e-12:
        y = (y - np.mean(y)) / np.std(y)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def segment_formula_features(segments: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in segments.columns if c.startswith("feature_")]
    rows = []
    for _, row in segments.iterrows():
        formula = str(row.get("subformula", ""))
        values = evaluate_formula(formula)
        first = safe_stat(values[:1], np.nanmean)
        last = safe_stat(values[-1:], np.nanmean)
        out = {
            "dataset": row.get("dataset"),
            "song_id": row.get("song_id"),
            "segment_id": row.get("segment_id"),
            "label": row.get("label"),
            "token": row.get("token"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration": row.get("duration"),
            "formula": formula,
            "formula_length": row.get("formula_length"),
            "formula_start_value": first,
            "formula_end_value": last,
            "formula_slope": last - first if np.isfinite(first) and np.isfinite(last) else np.nan,
            "formula_mean_value": safe_stat(values, np.nanmean),
            "formula_std_value": safe_stat(values, np.nanstd),
            "formula_min_value": safe_stat(values, np.nanmin),
            "formula_max_value": safe_stat(values, np.nanmax),
            "formula_bandwidth": safe_stat(values, np.nanmax) - safe_stat(values, np.nanmin),
            "formula_curvature": curvature(values),
            "formula_nonlinearity_rmse": linear_rmse(values),
        }
        out.update(formula_structure_features(formula))
        out.update(matrix_contour_features(row, feature_cols))
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate_recording_features(
    segment_features: pd.DataFrame, n_formula_clusters: int
) -> pd.DataFrame:
    id_cols = {"dataset", "song_id", "label", "token", "formula"}
    numeric_cols = [
        col
        for col in segment_features.select_dtypes(include=[np.number]).columns
        if col not in id_cols and col != "segment_id"
    ]

    working = segment_features.copy()
    cluster_feature_cols = [
        col
        for col in numeric_cols
        if col not in {"start", "end"} and not working[col].isna().all()
    ]
    n_clusters = min(n_formula_clusters, max(1, len(working)))
    if cluster_feature_cols and len(working) >= n_clusters:
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KMeans(n_clusters=n_clusters, random_state=0, n_init=20),
        )
        working["formula_cluster"] = pipe.fit_predict(working[cluster_feature_cols])
    else:
        working["formula_cluster"] = 0
        n_clusters = 1

    rows = []
    group_cols = ["dataset", "song_id", "label"]
    for keys, group in working.groupby(group_cols, dropna=False):
        dataset, song_id, label = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "song_id": song_id,
            "label": label,
            "n_segments": len(group),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())

        counts = group["formula_cluster"].value_counts(normalize=True)
        for cluster_id in range(n_clusters):
            row[f"formula_cluster_{cluster_id}_fraction"] = float(counts.get(cluster_id, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    labels = np.unique(y_true)
    clusters = np.unique(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if matrix.shape[1] < len(clusters):
        matrix = confusion_matrix(y_true, y_pred)
    cost = matrix.max() - matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(matrix[row_ind, col_ind].sum() / matrix.sum()), matrix


def evaluate_formula_features(
    recordings: pd.DataFrame, out_dir: Path, eval_n_init: int = 1
) -> pd.DataFrame:
    matrix_dir = out_dir / "formula_baseline_confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [
        col
        for col in recordings.select_dtypes(include=[np.number]).columns
        if col not in {"seed"}
    ]
    rows = []
    for task, datasets in DATASETS.items():
        subset = recordings[recordings["dataset"].isin(datasets)].copy()
        labels = sorted(subset["dataset"].unique())
        if len(labels) < 2 or subset.empty:
            continue
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        y_true = subset["dataset"].map(label_to_int).to_numpy(dtype=int)
        best: tuple[float, int, np.ndarray] | None = None
        for seed in SEEDS:
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                KMeans(n_clusters=len(labels), random_state=seed, n_init=eval_n_init),
            )
            y_pred = pipe.fit_predict(subset[feature_cols])
            acc, matrix = clustering_accuracy(y_true, y_pred)
            rows.append(
                {
                    "dataset": task,
                    "method": "FormulaFeatures+KMeans",
                    "seed": seed,
                    "accuracy": acc * 100.0,
                    "n_recordings": len(subset),
                    "n_classes": len(labels),
                    "n_init": eval_n_init,
                }
            )
            if best is None or acc > best[0]:
                best = (acc, seed, matrix)

        if best is not None:
            save_confusion_matrix(
                best[2],
                labels,
                matrix_dir / f"{task}_FormulaFeatures_KMeans_seed{best[1]}_confusion.png",
                title=f"{task} FormulaFeatures+KMeans",
            )
            pd.DataFrame(best[2], index=labels).to_csv(
                matrix_dir / f"{task}_FormulaFeatures_KMeans_seed{best[1]}_confusion.csv"
            )
    return pd.DataFrame(rows)


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), max(3, len(labels) * 0.7)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("True class")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([short_name(label) for label in labels])
    ax.set_xticks(range(matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def short_name(label: str) -> str:
    return label.replace("1017_plots47_", "").replace("_1017", "").replace("Curate", "")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(args.segments_csv)
    _ = pd.read_csv(args.sequences_csv)

    segment_features = segment_formula_features(segments)
    segment_features.to_csv(args.out_dir / "formula_segment_features.csv", index=False)

    recording_features = aggregate_recording_features(
        segment_features, n_formula_clusters=args.n_formula_clusters
    )
    recording_features.to_csv(args.out_dir / "formula_recording_features.csv", index=False)

    evaluation = evaluate_formula_features(
        recording_features, args.out_dir, eval_n_init=args.eval_n_init
    )
    evaluation.to_csv(args.out_dir / "formula_baseline_evaluation.csv", index=False)
    print(f"Wrote {args.out_dir / 'formula_segment_features.csv'}")
    print(f"Wrote {args.out_dir / 'formula_recording_features.csv'}")
    print(f"Wrote {args.out_dir / 'formula_baseline_evaluation.csv'}")


if __name__ == "__main__":
    main()

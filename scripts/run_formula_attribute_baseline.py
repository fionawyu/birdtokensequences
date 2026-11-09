#!/usr/bin/env python3
"""KMeans baseline on the ten segment attributes of Section 2.6.

Replaces the contour features of run_committee_formula_contour_baseline.py
with the attributes computed by scripts/segment_attributes.py (SymPy
derivatives on each formula's own fitted time points, physical units).
Everything else is kept identical so the two can be compared: the same
recording-level summary (mean, std, min, max of every attribute plus the
number of segments), the same tasks and KMeans seeds, median imputation,
standard scaling, n_init=1 per seed, and Hungarian-matched accuracy.

Usage:
    python scripts/run_formula_attribute_baseline.py \
        --attributes-csv results_segment_attributes/segment_attributes.csv \
        --out-dir results_formula_attribute_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_committee_formula_baseline import (  # noqa: E402
    DATASETS, REPO_ROOT, SEEDS, clustering_accuracy, save_confusion_matrix,
)
from segment_attributes import ATTRIBUTES  # noqa: E402

METHOD_NAME = "FormulaAttributes+KMeans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--attributes-csv", type=Path,
                        default=REPO_ROOT / "results_segment_attributes" / "segment_attributes.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "results_formula_attribute_baseline")
    parser.add_argument("--eval-n-init", type=int, default=1)
    return parser.parse_args()


def aggregate_recording_attributes(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, song_id, label), group in segments.groupby(
            ["dataset", "song_id", "label"], dropna=False):
        row: dict[str, object] = {"dataset": dataset, "song_id": song_id,
                                  "label": label, "n_segments": len(group)}
        for column in ATTRIBUTES:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(recordings: pd.DataFrame, out_dir: Path, eval_n_init: int) -> pd.DataFrame:
    matrix_dir = out_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = list(recordings.select_dtypes(include=[np.number]).columns)
    rows = []
    for task, datasets in DATASETS.items():
        subset = recordings[recordings["dataset"].isin(datasets)].copy()
        labels = sorted(subset["dataset"].unique())
        if len(labels) < 2:
            continue
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        y_true = subset["dataset"].map(label_to_int).to_numpy(dtype=int)
        best = None
        for seed in SEEDS:
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                KMeans(n_clusters=len(labels), random_state=seed, n_init=eval_n_init),
            )
            y_pred = pipe.fit_predict(subset[feature_cols])
            acc, matrix = clustering_accuracy(y_true, y_pred)
            rows.append({"dataset": task, "method": METHOD_NAME, "seed": seed,
                         "accuracy": acc * 100.0, "n_recordings": len(subset),
                         "n_classes": len(labels), "n_init": eval_n_init})
            if best is None or acc > best[0]:
                best = (acc, seed, matrix)
        save_confusion_matrix(best[2], labels,
                              matrix_dir / f"{task}_{METHOD_NAME}_seed{best[1]}_confusion.png",
                              title=f"{task} {METHOD_NAME}")
        pd.DataFrame(best[2], index=labels).to_csv(
            matrix_dir / f"{task}_{METHOD_NAME}_seed{best[1]}_confusion.csv")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(args.attributes_csv)
    n_invalid = int((~segments["valid"].astype(bool)).sum())
    if n_invalid:
        print(f"{n_invalid} segment(s) with an undefined derivative; their "
              f"attributes are NaN and are skipped in the recording summary")
    recordings = aggregate_recording_attributes(segments)
    recordings.to_csv(args.out_dir / "formula_attribute_recording_features.csv", index=False)
    evaluation = evaluate(recordings, args.out_dir, args.eval_n_init)
    evaluation.to_csv(args.out_dir / "formula_attribute_baseline_evaluation.csv", index=False)
    print(evaluation.groupby("dataset")["accuracy"].agg(["mean", "std", "min", "max"]).round(1))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()

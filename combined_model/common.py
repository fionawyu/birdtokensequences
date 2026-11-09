"""Shared plumbing for the combined (attribute + sequence) model deliverable.

New module — does not import from or alter any script under scripts/ in a way
that changes its outputs. Where logic is duplicated (k-medoids, Hungarian
accuracy, the six evaluation settings) it is copied rather than imported so
this module has no risk of mutating shared state in the existing pipeline.

Seed protocol note (see results/summary.md for the full explanation): the
published Sequence-model numbers in the manuscript are not "four random
seeds" — they are four seeds hand-selected from a scan of 10,000 seeds per
setting, documented in
paper_results/chapter1/figure1_results_overview/figure1_results_overview_per_dataset_long_error_seed_strategy.md.
The user was informed of this and chose to keep matching the published
numbers as the baseline. SELECTED_SEEDS below reproduces them exactly
(verified against the cached long-form CSV) and is used throughout Steps
A-C so the mandatory alpha=0 / alpha=1 / c_family=1 endpoint checks hold by
construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_CSV = REPO_ROOT / "diagnosis" / "chapter1_diagnosis_matfeat" / "segments.csv"
ORIGINAL_SEQUENCES_CSV = (
    REPO_ROOT / "diagnosis" / "chapter1_pitch_diagnostic" / "evaluation_inputs" / "original_token_sequences.csv"
)
PITCH_SEQUENCES_CSV = (
    REPO_ROOT / "diagnosis" / "chapter1_pitch_diagnostic" / "evaluation_inputs" / "token_pitch_sequences.csv"
)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DIST_DIR = RESULTS_DIR / "distance_matrices"

SILENCE_TOKENS = {"ZERO", "SILENCE"}

# The six evaluation settings, identical to scripts/run_committee_formula_baseline.py
# and scripts/plot_chapter1_results_overview.py (both verified to define the same dict).
DATASETS: dict[str, list[str]] = {
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

SETTING_LABELS = {
    "SegMix": "Easy - WTSP/Feebee segment-mixed",
    "NoteMix": "Easy - YWAR/YRWA note-order",
    "BCCH-var": "Medium - BCCH variants",
    "Species": "Medium - Species",
    "AllVar": "Hard - Species + BCCH variants",
    "YNCCDialect": "Hard - Yellow Cardinal dialects",
}

# Genuinely fixed, non-cherry-picked seeds used by the existing (already
# published, already reproducible) attribute/KMeans baseline in
# scripts/run_committee_formula_baseline.py. Used only for the A4
# clusterer-isolation diagnostic.
ATTRIBUTE_BASELINE_SEEDS = [42, 156, 12, 1]

# Reproduces the published table exactly (verified 2026-07-30 by re-running
# scripts/plot_chapter1_results_overview.py --recompute --selected-seeds-csv
# against the cached long CSV: identical to 1e-9). Source:
# paper_results/chapter1/figure1_results_overview/figure1_results_overview_per_dataset_long_error_selected_seeds.csv
SELECTED_SEEDS: dict[str, list[int]] = {
    "SegMix": [0, 1, 2, 3],
    "NoteMix": [0, 3, 4, 1],
    "BCCH-var": [1, 2, 4, 5],
    "Species": [0, 2, 3, 22],
    "AllVar": [12, 57, 4, 7],
    "YNCCDialect": [39, 58, 99, 36],
}


def kmedoids_fit(
    distance_matrix: np.ndarray, n_clusters: int, seed: int, max_iter: int = 100
) -> np.ndarray:
    """k-medoids on a precomputed distance matrix. Identical algorithm to
    scripts/plot_chapter1_results_overview.py:kmedoids_fit, reimplemented
    here (not imported) to keep this module standalone and to accept any
    distance matrix (edit distance, attribute distance, or a fused one)."""
    rng = np.random.default_rng(seed)
    medoid_indices = rng.choice(distance_matrix.shape[0], n_clusters, replace=False)

    labels = np.zeros(distance_matrix.shape[0], dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
        new_medoid_indices = medoid_indices.copy()
        changed = False

        for cluster_idx in range(n_clusters):
            members = np.flatnonzero(labels == cluster_idx)
            if len(members) == 0:
                continue

            best_member = int(members[0])
            best_cost = float("inf")
            cluster_distances = distance_matrix[np.ix_(members, members)]
            member_costs = cluster_distances.sum(axis=1)
            for member_idx, candidate in enumerate(members):
                cost = float(member_costs[member_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_member = int(candidate)

            if best_member != int(medoid_indices[cluster_idx]):
                new_medoid_indices[cluster_idx] = best_member
                changed = True

        medoid_indices = new_medoid_indices
        if not changed:
            break

    return labels


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    if matrix.shape[1] < len(clusters):
        matrix = confusion_matrix(y_true, y_pred)
    cost = matrix.max() - matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(matrix[row_ind, col_ind].sum() / matrix.sum())


def rank_normalise(distance_matrix: np.ndarray) -> np.ndarray:
    """Rank-normalise the upper triangle (excluding diagonal) to [0, 1] and
    mirror it, per the fusion protocol in Step B1."""
    from scipy.stats import rankdata

    n = distance_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    ranks = rankdata(distance_matrix[iu], method="average") / len(iu[0])
    out = np.zeros_like(distance_matrix, dtype=float)
    out[iu] = ranks
    return out + out.T


def load_labels(dataset_col: pd.Series, datasets: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted(set(dataset_col) & set(datasets))
    label_to_int = {label: idx for idx, label in enumerate(labels)}
    y_true = dataset_col.map(label_to_int).to_numpy(dtype=int)
    return y_true, label_to_int

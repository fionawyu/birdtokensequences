"""The five/six compared methods, each expressed as fit-on-train /
predict-on-test. Every function here receives the training indices and the
test indices and never receives the test labels.

Each returns a dict with:
    test_clusters   cluster id per held-out recording
    test_distance   distance from each held-out recording to the training
                    medoid/centroid it was assigned to
    all_distances   (n_test, k) distances to every training medoid/centroid
    train_clusters  cluster id per TRAINING recording (used only to build the
                    cluster->class map from training labels)
    model           what was fitted: medoid recording IDs, or centroids
"""
from __future__ import annotations

import numpy as np

from cv_protocol import (
    N_KMEANS_INIT, N_KMEDOIDS_RESTARTS, fit_kmedoids_training,
    predict_nearest_centroid, predict_nearest_medoid,
)

# --------------------------------------------------------------------------
# Levenshtein distance over token sequences -- identical rule to
# scripts/plot_chapter1_results_overview.py:edit_distance (unit insert, unit
# delete, unit substitution), just JIT-compiled so the large ARU pool is
# tractable. verify_edit_distance() below proves the two agree.
# --------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                        # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _edit_distance_codes(a, b):
    m = a.shape[0]
    n = b.shape[0]
    prev = np.empty(n + 1, dtype=np.float64)
    cur = np.empty(n + 1, dtype=np.float64)
    for j in range(n + 1):
        prev[j] = j
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            sub = 0.0 if a[i - 1] == b[j - 1] else 1.0
            d1 = prev[j] + 1.0
            d2 = cur[j - 1] + 1.0
            d3 = prev[j - 1] + sub
            best = d1
            if d2 < best:
                best = d2
            if d3 < best:
                best = d3
            cur[j] = best
        for j in range(n + 1):
            prev[j] = cur[j]
    return prev[n]


def encode_sequences(sequences: list[list[str]]) -> tuple[list[np.ndarray], dict[str, int]]:
    vocab: dict[str, int] = {}
    coded = []
    for seq in sequences:
        arr = np.empty(len(seq), dtype=np.int64)
        for i, tok in enumerate(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            arr[i] = vocab[tok]
        coded.append(arr)
    return coded, vocab


def edit_distance_matrix(sequences: list[list[str]]) -> np.ndarray:
    coded, _ = encode_sequences(sequences)
    n = len(coded)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(_edit_distance_codes(coded[i], coded[j]))
            out[i, j] = out[j, i] = d
    return out


def edit_distance_rect(rows: list[list[str]], cols: list[list[str]]) -> np.ndarray:
    """Distances from each `rows` sequence to each `cols` sequence -- the
    test-to-training-medoid step, which needs no full matrix."""
    coded, vocab = encode_sequences(list(rows) + list(cols))
    r = coded[: len(rows)]
    c = coded[len(rows):]
    out = np.zeros((len(r), len(c)), dtype=float)
    for i in range(len(r)):
        for j in range(len(c)):
            out[i, j] = float(_edit_distance_codes(r[i], c[j]))
    return out


def _reference_edit_distance(a: list[str], b: list[str]) -> float:
    """scripts/plot_chapter1_results_overview.py:edit_distance, verbatim."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=float)
    dp[:, 0] = np.arange(m + 1, dtype=float)
    dp[0, :] = np.arange(n + 1, dtype=float)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = 0.0 if a[i - 1] == b[j - 1] else 1.0
            dp[i, j] = min(dp[i - 1, j] + 1.0, dp[i, j - 1] + 1.0, dp[i - 1, j - 1] + sub)
    return float(dp[m, n])


def verify_edit_distance(sequences: list[list[str]], n_pairs: int = 400, seed: int = 0) -> int:
    """Check the JIT-compiled distance against the published implementation on
    randomly sampled pairs. Returns the number of pairs checked; raises on
    any disagreement."""
    rng = np.random.default_rng(seed)
    coded, _ = encode_sequences(sequences)
    n = len(sequences)
    checked = 0
    for _ in range(n_pairs):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        fast = float(_edit_distance_codes(coded[i], coded[j]))
        ref = _reference_edit_distance(sequences[i], sequences[j])
        if abs(fast - ref) > 1e-9:
            raise AssertionError(f"edit distance mismatch on pair ({i},{j}): {fast} vs {ref}")
        checked += 1
    return checked


# --------------------------------------------------------------------------
# Token order shuffling (the Shuffled control)
# --------------------------------------------------------------------------
def shuffle_sequences(sequences: list[list[str]], rep: int, shuffle_seed_base: int) -> list[list[str]]:
    """Shuffle each recording's token order independently, with a recorded
    per-recording seed. Composition is preserved, order is destroyed. Applied to
    training and held-out recordings alike, so the control is subject to exactly
    the same held-out protocol as Token."""
    out = []
    for idx, seq in enumerate(sequences):
        rng = np.random.default_rng(shuffle_seed_base + 1000 * rep + idx)
        arr = list(seq)
        rng.shuffle(arr)
        out.append(arr)
    return out


# --------------------------------------------------------------------------
# Distance-matrix (k-medoids) methods
# --------------------------------------------------------------------------
def run_kmedoids_from_full_matrix(full_D, train_idx, test_idx, k, base_seed,
                                  recording_ids, n_restarts=N_KMEDOIDS_RESTARTS):
    """Token / Shuffled / Trajectory-DTW / Perch+k-medoids.

    `full_D` is a pairwise distance matrix over the whole pool. Only its
    training block is used to fit, and only test-row/medoid-column entries are
    used to predict -- see the note in data_sources.py on why restricting a
    purely pairwise matrix is identical to rebuilding it from the training
    recordings (checked numerically in validate_cv.py)."""
    D_train = full_D[np.ix_(train_idx, train_idx)]
    train_clusters, medoids_local, cost, winning_seed = fit_kmedoids_training(
        D_train, k, base_seed, n_restarts
    )
    medoids_global = np.asarray(train_idx, dtype=int)[medoids_local]
    D_test_to_medoids = full_D[np.ix_(test_idx, medoids_global)]
    test_clusters, test_distance = predict_nearest_medoid(D_test_to_medoids)
    return {
        "train_clusters": train_clusters,
        "test_clusters": test_clusters,
        "test_distance": test_distance,
        "all_distances": D_test_to_medoids,
        "model": {
            "kind": "medoids",
            "medoid_recording_ids": [recording_ids[i] for i in medoids_global],
            "medoid_pool_indices": medoids_global.tolist(),
            "within_cluster_cost": cost,
            "winning_restart_seed": winning_seed,
            "n_restarts": n_restarts,
        },
    }


def run_kmedoids_from_vectors(X, train_idx, test_idx, k, base_seed, recording_ids,
                              metric="euclidean", n_restarts=N_KMEDOIDS_RESTARTS):
    """Perch v2 + k-medoids. Euclidean distance on raw (non-normalized)
    embeddings, matching the published protocol
    (paper_results/chapter1/method_comparison/perch2_euclidean_raw.csv)."""
    from scipy.spatial.distance import cdist
    D = cdist(X, X, metric=metric)
    return run_kmedoids_from_full_matrix(D, train_idx, test_idx, k, base_seed,
                                          recording_ids, n_restarts)


# --------------------------------------------------------------------------
# k-means methods
# --------------------------------------------------------------------------
def run_kmeans(X, train_idx, test_idx, k, seed, use_scaler: bool):
    """Formula attributes + k-means (with training-only median imputation and
    training-only standardization) and Perch v2 + k-means (raw embeddings, no
    scaler, matching the published protocol).

    The imputer and scaler see ONLY the training rows; the held-out rows are
    transformed with those training statistics."""
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    prep = {}
    if use_scaler:
        imputer = SimpleImputer(strategy="median").fit(X_train_raw)
        X_train = imputer.transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        prep = {
            "imputer_statistics_": imputer.statistics_.tolist(),
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
        }
    else:
        X_train, X_test = X_train_raw, X_test_raw

    km = KMeans(n_clusters=k, n_init=N_KMEANS_INIT, random_state=seed).fit(X_train)
    test_clusters, test_distance, all_d = predict_nearest_centroid(X_test, km.cluster_centers_)
    return {
        "train_clusters": km.labels_.astype(int),
        "test_clusters": test_clusters,
        "test_distance": test_distance,
        "all_distances": all_d,
        "model": {
            "kind": "centroids",
            "centroids": km.cluster_centers_,
            "inertia": float(km.inertia_),
            "kmeans_random_state": int(seed),
            "n_init": N_KMEANS_INIT,
            "preprocessing_fitted_on_training": prep,
        },
    }

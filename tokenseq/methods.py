"""The compared configurations (Sec. 3.2), all evaluated with the same splits.

Each builder returns a `classify.Method`: a function of
(train_idx, test_idx, y_train, run_seed) -> predicted test classes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import classify, distances, tokens
from .attributes import recording_vector
from .pipeline import sound_segments, token_sequence

METHODS = (
    "token", "shuffled", "segment_attributes", "meanfreq_tokens",
    "trajectory_dtw", "perch_kmeans", "perch_kmedoids",
)

LABELS = {
    "token": "Token + k-medoids",
    "shuffled": "Shuffled tokens + k-medoids",
    "segment_attributes": "Segment attributes + k-medoids",
    "meanfreq_tokens": "Mean frequency in tokens + k-medoids",
    "trajectory_dtw": "Trajectory-DTW + k-medoids",
    "perch_kmeans": "Perch v2 + k-means",
    "perch_kmedoids": "Perch v2 + k-medoids",
}


def _precomputed(D: np.ndarray, n_classes: int) -> classify.Method:
    def method(train, test, y_train, seed):
        return classify.kmedoids_classifier(D[np.ix_(train, train)], D[np.ix_(test, train)],
                                            y_train, n_classes, seed)
    return method


def token_method(records: list[dict], n_classes: int) -> classify.Method:
    """Token + k-medoids: edit distance between token sequences."""
    return _precomputed(distances.edit_distance_matrix([token_sequence(r) for r in records]),
                        n_classes)


def shuffled_method(records: list[dict], n_classes: int) -> classify.Method:
    """Shuffled tokens + k-medoids: each recording's sequence is randomly
    permuted (a new permutation in every run), keeping its tokens."""
    seqs = [token_sequence(r) for r in records]

    @lru_cache(maxsize=None)
    def matrix(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return distances.edit_distance_matrix([list(rng.permutation(s)) for s in seqs])

    def method(train, test, y_train, seed):
        D = matrix(seed)
        return classify.kmedoids_classifier(D[np.ix_(train, train)], D[np.ix_(test, train)],
                                            y_train, n_classes, seed)
    return method


def attribute_matrix(records: list[dict]) -> np.ndarray:
    return np.stack([recording_vector([s["attributes"] for s in sound_segments(r)])
                     for r in records])


def attributes_method(records: list[dict], n_classes: int) -> classify.Method:
    """Segment attributes + k-medoids: 41-dim vectors, standardised with
    training-fold statistics, Euclidean distance."""
    X = attribute_matrix(records)

    def method(train, test, y_train, seed):
        mu = np.nanmean(X[train], axis=0)
        sd = np.nanstd(X[train], axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = np.where(np.isfinite(X), X, mu)
        Z = (Z - mu) / sd
        return classify.kmedoids_classifier(distances.euclidean(Z[train], Z[train]),
                                            distances.euclidean(Z[test], Z[train]),
                                            y_train, n_classes, seed)
    return method


def meanfreq_sequences(records: list[dict]) -> list[list[str]]:
    """Replace each shape token by <tertile>_<shape>, with tertiles of the
    segment mean frequency computed over all sound segments of the task."""
    means = [s["attributes"]["mean_khz"] for r in records for s in sound_segments(r)
             if s["token"] in tokens.SHAPE_TOKENS]
    q1, q2 = np.quantile(means, [1 / 3, 2 / 3])
    out = []
    for r in records:
        seq = []
        for iv in r["intervals"]:
            if iv["kind"] == "sound" and iv["token"] in tokens.SHAPE_TOKENS:
                m = iv["attributes"]["mean_khz"]
                seq.append(f"{'LOW' if m <= q1 else 'MEDIUM' if m <= q2 else 'HIGH'}_{iv['token']}")
            else:
                seq.append(iv["token"])
        out.append(seq)
    return out


def meanfreq_method(records: list[dict], n_classes: int) -> classify.Method:
    return _precomputed(distances.edit_distance_matrix(meanfreq_sequences(records)), n_classes)


def dtw_method(records: list[dict], n_classes: int) -> classify.Method:
    """Trajectory-DTW + k-medoids on the dominant-frequency trajectories (kHz)
    of the sound segments, concatenated in order."""
    series = [np.concatenate([s["trajectory_khz"] for s in sound_segments(r)] or [[0.0]])
              for r in records]
    return _precomputed(distances.dtw_matrix(series), n_classes)


def perch_kmeans_method(E: np.ndarray, n_classes: int, n_init: int) -> classify.Method:
    def method(train, test, y_train, seed):
        return classify.kmeans_classifier(E[train], E[test], y_train, n_classes, seed, n_init)
    return method


def perch_kmedoids_method(E: np.ndarray, n_classes: int) -> classify.Method:
    return _precomputed(distances.euclidean(E, E), n_classes)


def build(name: str, records: list[dict], n_classes: int, cfg: dict,
          embeddings: np.ndarray | None = None) -> classify.Method:
    if name == "token":
        return token_method(records, n_classes)
    if name == "shuffled":
        return shuffled_method(records, n_classes)
    if name == "segment_attributes":
        return attributes_method(records, n_classes)
    if name == "meanfreq_tokens":
        return meanfreq_method(records, n_classes)
    if name == "trajectory_dtw":
        return dtw_method(records, n_classes)
    if embeddings is None:
        raise ValueError(f"{name} needs Perch v2 embeddings (--embeddings)")
    if name == "perch_kmeans":
        return perch_kmeans_method(embeddings, n_classes, cfg["evaluation"]["kmeans_n_init"])
    if name == "perch_kmedoids":
        return perch_kmedoids_method(embeddings, n_classes)
    raise ValueError(f"unknown method {name}")

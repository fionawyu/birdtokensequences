"""Clustering-based classifiers and cross-validation (Sec. 2.8).

Training: cluster the training recordings into k = number of classes groups
(labels are not given to the clustering algorithm), then pair clusters with
classes by the Hungarian algorithm so that the number of correctly matched
training recordings is maximal.
Test: assign each test recording to the class of its nearest medoid/centroid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


def hungarian_map(clusters: np.ndarray, labels: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    """cluster index -> class index maximising matched training recordings."""
    counts = np.zeros((k, n_classes), dtype=int)
    np.add.at(counts, (clusters, labels), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.full(k, -1, dtype=int)
    mapping[rows] = cols
    return mapping


def kmedoids_classifier(D_train: np.ndarray, D_test_train: np.ndarray, y_train: np.ndarray,
                        n_classes: int, seed: int) -> np.ndarray:
    import kmedoids

    res = kmedoids.fasterpam(D_train, n_classes, max_iter=100, init="random", random_state=seed)
    medoids = np.asarray(res.medoids)
    clusters = np.argmin(D_train[:, medoids], axis=1)
    mapping = hungarian_map(clusters, y_train, n_classes, n_classes)
    return mapping[np.argmin(D_test_train[:, medoids], axis=1)]


def kmeans_classifier(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                      n_classes: int, seed: int, n_init: int = 10) -> np.ndarray:
    km = KMeans(n_clusters=n_classes, n_init=n_init, random_state=seed).fit(X_train)
    mapping = hungarian_map(km.labels_, y_train, n_classes, n_classes)
    return mapping[km.predict(X_test)]


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------

# A method maps (train_idx, test_idx, run_seed) to predicted class indices of
# the test recordings, using the training labels only.
Method = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class CVResult:
    run_accuracies: list[float] = field(default_factory=list)
    predictions: list[np.ndarray] = field(default_factory=list)   # one array per run

    def summary(self) -> dict:
        a = np.asarray(self.run_accuracies) * 100
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                "max": float(a.max()), "runs": a.round(4).tolist()}


def cross_validate(method: Method, y: np.ndarray, run_seeds: list[int], n_folds: int) -> CVResult:
    """Repeated stratified k-fold CV. Each run pools the predictions of its
    folds into one accuracy; every recording is tested once per run."""
    y = np.asarray(y)
    out = CVResult()
    for seed in run_seeds:
        pred = np.full(y.shape, -1)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train, test in skf.split(np.zeros(len(y)), y):
            assert not set(train) & set(test)
            pred[test] = method(train, test, y[train], seed)
        assert np.all(pred >= 0)
        out.run_accuracies.append(float(np.mean(pred == y)))
        out.predictions.append(pred)
    return out


def confusion(y: np.ndarray, predictions: list[np.ndarray], n_classes: int) -> np.ndarray:
    """Confusion counts summed over runs (rows: true class, columns: assigned)."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for p in predictions:
        np.add.at(M, (y, p), 1)
    return M

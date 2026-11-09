"""Held-out cross-validation protocol shared by every method in this directory.

Design contract (enforced by the guards below, not merely by convention):

  1. Splits are made once per (dataset, repetition, fold) over RECORDING IDs and
     handed to every method unchanged, so all methods are scored on identical
     held-out recordings.
  2. Everything that is fitted -- distance matrices used for clustering,
     medoids, centroids, imputer medians, scaler mean/scale, and the
     cluster->class mapping -- is derived from TRAINING recordings only.
  3. Test labels are wrapped in `SealedLabels` and are unreadable until
     `unseal_for_scoring()` is called, which happens exactly once per fold,
     after the predictions for that fold already exist. Any earlier read
     raises LabelLeakageError instead of silently returning the labels.

Nothing in this module imports from, writes to, or otherwise mutates the
existing pipeline; the k-medoids algorithm is copied from
combined_model.common.kmedoids_fit (same assign/update rule, same
np.random.default_rng seeding) so that held-out numbers are comparable to the
published whole-dataset numbers rather than reflecting a different clusterer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------
# Fixed, recorded random seeds. Everything stochastic in this directory is
# derived deterministically from these three constants.
# --------------------------------------------------------------------------
N_REPETITIONS = 4
MAX_FOLDS = 5
SPLIT_SEEDS = [20260901, 20260902, 20260903, 20260904]   # one per repetition
CLUSTERING_SEED_BASE = 700_000                            # k-medoids / k-means restarts
SHUFFLE_SEED_BASE = 900_000                               # token-order shuffling
N_KMEDOIDS_RESTARTS = 25
N_KMEANS_INIT = 10

assert len(SPLIT_SEEDS) == N_REPETITIONS


def seeds_manifest() -> dict:
    return {
        "n_repetitions": N_REPETITIONS,
        "max_folds": MAX_FOLDS,
        "split_seeds_per_repetition": SPLIT_SEEDS,
        "clustering_seed_base": CLUSTERING_SEED_BASE,
        "shuffle_seed_base": SHUFFLE_SEED_BASE,
        "n_kmedoids_restarts": N_KMEDOIDS_RESTARTS,
        "n_kmeans_init": N_KMEANS_INIT,
        "fold_clustering_seed_rule": "CLUSTERING_SEED_BASE + 1000*rep + 100*fold + restart",
        "recording_shuffle_seed_rule": "SHUFFLE_SEED_BASE + 1000*rep + index_of_recording_in_pool",
    }


# --------------------------------------------------------------------------
# Test-label seal
# --------------------------------------------------------------------------
class LabelLeakageError(RuntimeError):
    pass


class SealedLabels:
    """Holds held-out labels and refuses to reveal them until scoring time."""

    def __init__(self, y_test: np.ndarray, tag: str):
        self._y = np.asarray(y_test)
        self.tag = tag
        self.sealed = True
        self.access_log: list[str] = []

    def __len__(self) -> int:          # length is not label information
        return len(self._y)

    def unseal_for_scoring(self, reason: str) -> np.ndarray:
        self.sealed = False
        self.access_log.append(reason)
        return self._y

    def values(self) -> np.ndarray:
        if self.sealed:
            raise LabelLeakageError(
                f"held-out labels for {self.tag} were read before scoring "
                f"(unseal_for_scoring has not been called)"
            )
        self.access_log.append("values()")
        return self._y

    # Deliberately hostile to accidental use in fitting code.
    def __array__(self, *args, **kwargs):
        raise LabelLeakageError(f"held-out labels for {self.tag} coerced to array before scoring")

    def __iter__(self):
        raise LabelLeakageError(f"held-out labels for {self.tag} iterated before scoring")

    def __getitem__(self, item):
        raise LabelLeakageError(f"held-out labels for {self.tag} indexed before scoring")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
@dataclass
class Fold:
    dataset: str
    rep: int
    fold: int
    split_seed: int
    n_splits: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    recording_ids: list[str] = field(repr=False, default_factory=list)

    @property
    def train_ids(self) -> list[str]:
        return [self.recording_ids[i] for i in self.train_idx]

    @property
    def test_ids(self) -> list[str]:
        return [self.recording_ids[i] for i in self.test_idx]

    @property
    def clustering_seed(self) -> int:
        return CLUSTERING_SEED_BASE + 1000 * self.rep + 100 * self.fold


def choose_n_splits(y: np.ndarray, max_folds: int = MAX_FOLDS) -> int:
    """5 folds, unless the smallest class has fewer than 5 recordings, in which
    case the fold count drops to the smallest class size (>=2)."""
    smallest = int(np.min(np.bincount(np.asarray(y, dtype=int))))
    return max(2, min(max_folds, smallest))


def make_folds(dataset: str, y: np.ndarray, recording_ids: list[str]) -> list[Fold]:
    """Stratified k-fold over recording IDs, repeated N_REPETITIONS times with
    the fixed seeds above. The returned folds are the ONLY splits any method in
    this directory is allowed to use."""
    y = np.asarray(y, dtype=int)
    if len(y) != len(recording_ids):
        raise ValueError("y and recording_ids length mismatch")
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError(f"{dataset}: duplicate recording IDs in the evaluation pool")

    n_splits = choose_n_splits(y)
    folds: list[Fold] = []
    for rep, split_seed in enumerate(SPLIT_SEEDS):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            folds.append(
                Fold(dataset=dataset, rep=rep, fold=fold, split_seed=split_seed,
                     n_splits=n_splits, train_idx=tr, test_idx=te,
                     recording_ids=list(recording_ids))
            )
    validate_folds(folds, recording_ids)
    return folds


def validate_folds(folds: list[Fold], recording_ids: list[str]) -> list[str]:
    """Check 1: disjointness and exhaustiveness of every split. Returns the
    lines of the validation report; raises on any violation."""
    lines = []
    all_ids = set(recording_ids)
    by_rep: dict[int, list[Fold]] = {}
    for f in folds:
        by_rep.setdefault(f.rep, []).append(f)

    for rep, reps_folds in sorted(by_rep.items()):
        seen_test: list[str] = []
        for f in reps_folds:
            tr, te = set(f.train_ids), set(f.test_ids)
            overlap = tr & te
            if overlap:
                raise AssertionError(
                    f"{f.dataset} rep{rep} fold{f.fold}: {len(overlap)} recording ID(s) "
                    f"in BOTH train and test, e.g. {sorted(overlap)[:3]}"
                )
            if tr | te != all_ids:
                raise AssertionError(f"{f.dataset} rep{rep} fold{f.fold}: train+test != pool")
            if len(tr) + len(te) != len(all_ids):
                raise AssertionError(f"{f.dataset} rep{rep} fold{f.fold}: train/test sizes do not sum to n")
            seen_test.extend(f.test_ids)
        if sorted(seen_test) != sorted(all_ids):
            raise AssertionError(f"{f.dataset} rep{rep}: recordings not tested exactly once")
        lines.append(
            f"  {folds[0].dataset:<14s} rep{rep}: {len(reps_folds)} folds, "
            f"train/test disjoint in every fold, every one of {len(all_ids)} recordings "
            f"held out exactly once  [OK]"
        )
    return lines


def assert_identical_splits(fold_sets: dict[str, list[Fold]]) -> list[str]:
    """Check 2: every method received byte-identical splits."""
    names = list(fold_sets)
    ref_name = names[0]
    ref = fold_sets[ref_name]
    lines = []
    for name in names[1:]:
        other = fold_sets[name]
        if len(other) != len(ref):
            raise AssertionError(f"{name} has {len(other)} folds, {ref_name} has {len(ref)}")
        for a, b in zip(ref, other):
            if a.rep != b.rep or a.fold != b.fold:
                raise AssertionError(f"{name}: fold ordering differs from {ref_name}")
            if a.train_ids != b.train_ids or a.test_ids != b.test_ids:
                raise AssertionError(
                    f"{name}: rep{a.rep} fold{a.fold} split differs from {ref_name}"
                )
    lines.append(f"  all {len(names)} methods use identical splits ({len(ref)} folds)  [OK]")
    return lines


# --------------------------------------------------------------------------
# Clustering fitted on training recordings only
# --------------------------------------------------------------------------
def _kmedoids_once(D: np.ndarray, k: int, seed: int, max_iter: int = 100):
    """Same algorithm as combined_model.common.kmedoids_fit, additionally
    returning the medoid indices and the total within-cluster cost so a restart
    can be selected WITHOUT looking at any label."""
    rng = np.random.default_rng(seed)
    medoids = rng.choice(D.shape[0], k, replace=False)
    labels = np.zeros(D.shape[0], dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        changed = False
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members) == 0:
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            best = int(members[int(np.argmin(costs))])
            if best != int(medoids[c]):
                new_medoids[c] = best
                changed = True
        medoids = new_medoids
        if not changed:
            break
    labels = np.argmin(D[:, medoids], axis=1)
    cost = float(D[np.arange(D.shape[0]), medoids[labels]].sum())
    return labels, np.asarray(medoids, dtype=int), cost


def fit_kmedoids_training(D_train: np.ndarray, k: int, base_seed: int,
                          n_restarts: int = N_KMEDOIDS_RESTARTS):
    """Fit k-medoids on the training distance matrix. The restart that is kept
    is the one with the lowest total within-cluster distance -- an unsupervised
    criterion, so no label (train or test) influences model selection."""
    best = None
    for r in range(n_restarts):
        labels, medoids, cost = _kmedoids_once(D_train, k, base_seed + r)
        if best is None or cost < best[2]:
            best = (labels, medoids, cost, base_seed + r)
    return best  # labels, medoid_local_indices, cost, winning_seed


def predict_nearest_medoid(D_test_to_medoids: np.ndarray):
    """Assign each held-out recording to the nearest TRAINING medoid."""
    clusters = np.argmin(D_test_to_medoids, axis=1)
    dists = D_test_to_medoids[np.arange(len(clusters)), clusters]
    return clusters.astype(int), dists.astype(float)


def predict_nearest_centroid(X_test: np.ndarray, centroids: np.ndarray):
    """Assign each held-out recording to the nearest TRAINING centroid
    (Euclidean, identical to sklearn KMeans.predict)."""
    d = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)
    clusters = np.argmin(d, axis=1)
    return clusters.astype(int), d[np.arange(len(clusters)), clusters].astype(float), d


# --------------------------------------------------------------------------
# Cluster -> class mapping, from training labels only
# --------------------------------------------------------------------------
def hungarian_map_from_training(train_clusters: np.ndarray, y_train: np.ndarray,
                                k: int, n_classes: int) -> dict[int, int]:
    """Optimal one-to-one cluster->class assignment maximizing agreement on the
    TRAINING recordings (Hungarian algorithm). Clusters that win no class (only
    possible when k != n_classes) map to -1, which always scores as incorrect."""
    contingency = np.zeros((k, n_classes), dtype=np.int64)
    for cluster, cls in zip(np.asarray(train_clusters, dtype=int), np.asarray(y_train, dtype=int)):
        contingency[cluster, cls] += 1
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {c: -1 for c in range(k)}
    for r, c in zip(rows, cols):
        mapping[int(r)] = int(c)
    return mapping


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def summarize_repetitions(rep_accuracies: list[float]) -> dict:
    a = np.asarray(rep_accuracies, dtype=float)
    return {
        "n_repetitions": int(a.size),
        "mean_accuracy": float(np.mean(a)),
        "sd_accuracy": float(np.std(a, ddof=0)),
        "median_accuracy": float(np.median(a)),
        "best_accuracy": float(np.max(a)),
        "min_accuracy": float(np.min(a)),
    }


def write_seeds_manifest(out_dir: Path, extra: dict | None = None) -> None:
    payload = seeds_manifest()
    if extra:
        payload.update(extra)
    (out_dir / "seeds.json").write_text(json.dumps(payload, indent=2) + "\n")

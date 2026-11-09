"""Pairwise distances between recordings (Sec. 2.7, 3.2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from rapidfuzz.distance import Levenshtein as _Lev
except ImportError:  # pragma: no cover - optional speed-up
    _Lev = None


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Minimum number of insertions, deletions and substitutions turning a into b."""
    if _Lev is not None:
        return int(_Lev.distance(list(a), list(b)))
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_distance_matrix(seqs: list[list[str]]) -> np.ndarray:
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = levenshtein(seqs[i], seqs[j])
    return D


def dtw_matrix(series: list[np.ndarray]) -> np.ndarray:
    """DTW distance (dtaidistance; squared-difference cost) between trajectories."""
    from dtaidistance import dtw

    s = [np.asarray(x, dtype=np.double) for x in series]
    try:
        D = dtw.distance_matrix_fast(s)
    except Exception:  # C extension unavailable
        D = dtw.distance_matrix(s, use_c=False)
    D = np.asarray(D, dtype=float)
    D = np.where(np.isfinite(D), D, 0.0)
    return np.maximum(D, D.T)


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of a and the rows of b."""
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))

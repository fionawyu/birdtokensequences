"""Token-sequence measures (Sec. 3.3.1, Appendix D.2-D.3)."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.where(norms > 0, norms, 1.0)
    return np.clip(1.0 - U @ U.T, 0.0, 2.0)


def within_token_distance(vectors: np.ndarray) -> tuple[float, int]:
    """D_j: mean cosine distance from the medoid to the other segment vectors
    assigned to token j. Returns (D_j, medoid index); nan for < 2 segments."""
    if len(vectors) < 2:
        return float("nan"), 0
    D = cosine_distance_matrix(np.asarray(vectors, dtype=float))
    m = int(np.argmin(D.sum(axis=1)))
    return float(D[m].sum() / (len(vectors) - 1)), m


def class_within_token_distance(segment_vectors: list[np.ndarray], segment_tokens: list[str]) -> dict:
    """Per-token D_j for one class and their average over observed tokens."""
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for v, tok in zip(segment_vectors, segment_tokens):
        groups[tok].append(v)
    per_token = {tok: within_token_distance(np.stack(vs))[0] for tok, vs in groups.items()}
    vals = [v for v in per_token.values() if np.isfinite(v)]
    return {"per_token": per_token, "mean": float(np.mean(vals)) if vals else float("nan")}


def transition_entropy(seq: list[str]) -> float:
    """H = -sum_x p(x) sum_y p(y|x) log p(y|x), from adjacent token pairs."""
    pairs = list(zip(seq[:-1], seq[1:]))
    if not pairs:
        return float("nan")
    from_counts = Counter(x for x, _ in pairs)
    pair_counts = Counter(pairs)
    n = len(pairs)
    H = 0.0
    for (x, _y), c in pair_counts.items():
        p_x = from_counts[x] / n
        p_yx = c / from_counts[x]
        H -= p_x * p_yx * np.log(p_yx)
    return float(H)


def entropy_rate_gap(seq: list[str], n_permutations: int, rng: np.random.Generator) -> float:
    """Delta H = mean H over random permutations of seq - H of seq.
    Undefined (nan) for sequences with fewer than two tokens."""
    if len(seq) < 2:
        return float("nan")
    h_obs = transition_entropy(seq)
    h_perm = [transition_entropy(list(rng.permutation(seq))) for _ in range(n_permutations)]
    return float(np.mean(h_perm) - h_obs)

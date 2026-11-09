"""Symbolic regression of a dominant-frequency trajectory (Sec. 2.4).

A (1 + 19) genetic algorithm: the highest-fitness formula is the parent, and
19 offspring are produced from it by mutation in every generation. The parent
is replaced only when an offspring has higher fitness.

fitness = -( mean_i |f(t_i) - y_i|  +  size_penalty * log(size(f)) )

where y_i is the dominant frequency of frame i (as a position in the
band-limited spectrogram, scaled to [0, 1]) and t_i = i * frame_s * time_scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import formula as F


@dataclass
class FitResult:
    formula: str
    loss: float
    size: int


# --------------------------------------------------------------------------
# random formulas and mutation
# --------------------------------------------------------------------------

def random_constant(rng: np.random.Generator, const_range) -> str:
    lo, hi = const_range
    return f"{round(float(rng.uniform(lo, hi)), 3):g}"


def random_formula(rng: np.random.Generator, max_nodes: int, const_range) -> F.Node:
    """Draw a formula with at most `max_nodes` nodes from the grammar.

    At each node the production (terminal, unary or binary) is chosen
    uniformly among those that fit in the remaining node budget.
    """
    kinds = ["T"] + (["U"] if max_nodes >= 2 else []) + (["B"] if max_nodes >= 3 else [])
    kind = kinds[rng.integers(len(kinds))]
    if kind == "T":
        return F.VAR if rng.random() < 0.5 else random_constant(rng, const_range)
    if kind == "U":
        op = F.UNARY[rng.integers(len(F.UNARY))]
        return (op, random_formula(rng, max_nodes - 1, const_range))
    op = F.BINARY[rng.integers(len(F.BINARY))]
    left = random_formula(rng, int(rng.integers(1, max_nodes - 1)), const_range)
    right = random_formula(rng, max_nodes - 1 - F.size(left), const_range)
    return (op, left, right)


def n_mutations(rng: np.random.Generator, mean: float) -> int:
    """Number of mutation operations, k = floor(X) + 1 with X ~ Exp(mean)."""
    return 1 if mean <= 0 else int(rng.exponential(mean)) + 1


def mutate(node: F.Node, rng: np.random.Generator, cfg: dict) -> F.Node:
    """Apply k subtree replacements; each replaces a uniformly chosen node with
    a new random subformula of at most `max_subformula_nodes` nodes, subject to
    the total limit of `max_formula_nodes` nodes."""
    for _ in range(n_mutations(rng, cfg["mutation_mean"])):
        paths = F.subtrees(node)
        for _attempt in range(1000):
            path = paths[rng.integers(len(paths))]
            new = random_formula(rng, cfg["max_subformula_nodes"], cfg["const_range"])
            cand = F.replace(node, path, new)
            if F.size(cand) <= cfg["max_formula_nodes"] and cand != node:
                node = cand
                break
    return node


# --------------------------------------------------------------------------
# loss and search
# --------------------------------------------------------------------------

def loss(node: F.Node, t: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    if cfg["derivative_guard"] and not F.derivatives_finite(node, t):
        return float("inf")
    pred = F.evaluate(node, t)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return float(np.mean(np.abs(pred - y)) + cfg["size_penalty"] * np.log(F.size(node)))


def fit(t: np.ndarray, y: np.ndarray, cfg: dict, rng: np.random.Generator) -> FitResult:
    """Fit one formula to the trajectory y(t)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # initial population; inadmissible candidates are regenerated
    pop: list[F.Node] = []
    while len(pop) < cfg["population_size"]:
        cand = random_formula(rng, cfg["max_formula_nodes"], cfg["const_range"])
        if not cfg["derivative_guard"] or F.derivatives_finite(cand, t):
            pop.append(cand)
    losses = [loss(p, t, y, cfg) for p in pop]
    best = int(np.argmin(losses))
    parent, parent_loss = pop[best], losses[best]

    for _ in range(cfg["generations"]):
        children = [mutate(parent, rng, cfg) for _ in range(cfg["population_size"] - 1)]
        child_losses = [loss(c, t, y, cfg) for c in children]   # inadmissible -> inf
        k = int(np.argmin(child_losses))
        if child_losses[k] < parent_loss:
            parent, parent_loss = children[k], child_losses[k]

    return FitResult(formula=F.to_string(parent), loss=parent_loss, size=F.size(parent))

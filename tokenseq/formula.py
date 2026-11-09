"""Formula space, evaluation and differentiation (Sec. 2.4, Appendix B).

A formula is a tree in prefix (Lisp) notation, e.g. ``(+ (sin t) 2)``:

    F -> T | U | B
    T -> t | C                         C ~ U[0, 10]
    U -> sqrt F | sqr F | exp F | log F | sin F | cos F | tan F | neg F
    B -> F + F | F - F | F * F | F / F

In memory a formula is either a terminal string ("t" or a numeric literal) or
a tuple ``(op, child, ...)``.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import sympy as sp

Node = Union[str, tuple]

UNARY = ("sqrt", "sqr", "exp", "log", "sin", "cos", "tan", "neg")
BINARY = ("+", "-", "*", "/")
VAR = "t"


# --------------------------------------------------------------------------
# parsing / printing
# --------------------------------------------------------------------------

def parse(text: str) -> Node:
    toks = text.replace("(", " ( ").replace(")", " ) ").split()
    if not toks:
        raise ValueError("empty formula")

    def rec(i: int) -> tuple[Node, int]:
        if toks[i] != "(":
            return toks[i], i + 1
        op, i = toks[i + 1], i + 2
        args = []
        while toks[i] != ")":
            a, i = rec(i)
            args.append(a)
        arity = 1 if op in UNARY else 2 if op in BINARY else -1
        if arity != len(args):
            raise ValueError(f"bad operator or arity: {op}")
        return (op, *args), i + 1

    node, i = rec(0)
    if i != len(toks):
        raise ValueError(f"trailing tokens in {text!r}")
    return node


def to_string(node: Node) -> str:
    if isinstance(node, str):
        return node
    return "(" + " ".join([node[0]] + [to_string(c) for c in node[1:]]) + ")"


def to_infix(node: Node) -> str:
    """Human-readable infix form, e.g. ``sin(t) + 2``."""
    if isinstance(node, str):
        return node
    op, args = node[0], [to_infix(c) for c in node[1:]]
    if op in BINARY:
        return f"({args[0]} {op} {args[1]})"
    if op == "neg":
        return f"-({args[0]})"
    if op == "sqr":
        return f"({args[0]})^2"
    return f"{op}({args[0]})"


def size(node: Node) -> int:
    """Number of nodes (variables, constants and operators)."""
    if isinstance(node, str):
        return 1
    return 1 + sum(size(c) for c in node[1:])


def has_variable(node: Node) -> bool:
    if isinstance(node, str):
        return node == VAR
    return any(has_variable(c) for c in node[1:])


def subtrees(node: Node, path: tuple = ()) -> list[tuple]:
    """Paths (tuples of child indices) of all nodes in pre-order."""
    out = [path]
    if not isinstance(node, str):
        for k, c in enumerate(node[1:], start=1):
            out.extend(subtrees(c, path + (k,)))
    return out


def replace(node: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    k = path[0]
    children = list(node)
    children[k] = replace(node[k], path[1:], new)
    return tuple(children)


# --------------------------------------------------------------------------
# numeric evaluation
# --------------------------------------------------------------------------

def evaluate(node: Node, t: np.ndarray) -> np.ndarray:
    """Evaluate the formula at the time points `t` (non-finite values allowed)."""
    t = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        return np.broadcast_to(_eval(node, t), t.shape).astype(float)


def _eval(node: Node, t: np.ndarray) -> np.ndarray:
    if isinstance(node, str):
        return t if node == VAR else np.full_like(t, float(node))
    op = node[0]
    if op in UNARY:
        v = _eval(node[1], t)
        if op == "sqrt":
            return np.where(v >= 0, np.sqrt(np.abs(v)), np.nan)
        if op == "sqr":
            return v * v
        if op == "exp":
            return np.exp(v)
        if op == "log":
            return np.where(v > 0, np.log(np.abs(v)), np.nan)
        if op == "sin":
            return np.sin(v)
        if op == "cos":
            return np.cos(v)
        if op == "tan":
            return np.tan(v)
        return -v
    a, b = _eval(node[1], t), _eval(node[2], t)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)


def jet(node: Node, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, f', f'') at `t` by second-order forward-mode differentiation.

    Points where an operator or its derivative is undefined (sqrt/log of a
    non-positive argument, division by zero, overflow) come out non-finite.
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        return _jet(node, t)


def _jet(node: Node, t: np.ndarray):
    zeros = np.zeros_like(t)
    if isinstance(node, str):
        if node == VAR:
            return t, np.ones_like(t), zeros
        return np.full_like(t, float(node)), zeros, zeros
    op = node[0]
    if op in UNARY:
        v, d, dd = _jet(node[1], t)
        if op == "sqr":
            return v * v, 2 * v * d, 2 * d * d + 2 * v * dd
        if op == "neg":
            return -v, -d, -dd
        if op == "exp":
            e = np.exp(v)
            return e, e * d, e * (d * d + dd)
        if op == "sin":
            s, c = np.sin(v), np.cos(v)
            return s, c * d, -s * d * d + c * dd
        if op == "cos":
            s, c = np.sin(v), np.cos(v)
            return c, -s * d, -c * d * d - s * dd
        if op == "tan":
            tv = np.tan(v)
            s2 = 1 + tv * tv
            return tv, s2 * d, 2 * tv * s2 * d * d + s2 * dd
        if op == "sqrt":
            v = np.where(v > 0, v, np.nan)
            r = np.sqrt(v)
            return r, d / (2 * r), -d * d / (4 * v * r) + dd / (2 * r)
        v = np.where(v > 0, v, np.nan)  # log
        return np.log(v), d / v, -d * d / (v * v) + dd / v
    (a, ad, add), (b, bd, bdd) = _jet(node[1], t), _jet(node[2], t)
    if op == "+":
        return a + b, ad + bd, add + bdd
    if op == "-":
        return a - b, ad - bd, add - bdd
    if op == "*":
        return a * b, ad * b + a * bd, add * b + 2 * ad * bd + a * bdd
    b = np.where(b != 0, b, np.nan)
    v = a / b
    d = (ad - v * bd) / b
    return v, d, (add - 2 * d * bd - v * bdd) / b


def derivatives_finite(node: Node, t: np.ndarray) -> bool:
    """True if f, f' and f'' are finite and real at every point of `t`."""
    return all(np.all(np.isfinite(x)) for x in jet(node, t))


# --------------------------------------------------------------------------
# symbolic form (used for the slope and curvature attributes, Sec. 2.6)
# --------------------------------------------------------------------------

T_SYM = sp.Symbol("t", real=True)

_SYMPY_UNARY = {
    "sqrt": sp.sqrt, "sqr": lambda x: x ** 2, "exp": sp.exp, "log": sp.log,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "neg": lambda x: -x,
}


def to_sympy(node: Node) -> sp.Expr:
    if isinstance(node, str):
        return T_SYM if node == VAR else sp.Float(node)
    op = node[0]
    if op in UNARY:
        return _SYMPY_UNARY[op](to_sympy(node[1]))
    a, b = to_sympy(node[1]), to_sympy(node[2])
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]


def sympy_derivatives(node: Node, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, f', f'') at `t` using SymPy differentiation."""
    expr = to_sympy(node)
    d1 = sp.diff(expr, T_SYM)
    d2 = sp.diff(d1, T_SYM)
    t = np.asarray(t, dtype=float)
    out = []
    with np.errstate(all="ignore"):
        for e in (expr, d1, d2):
            f = sp.lambdify(T_SYM, e, modules="numpy")
            out.append(np.broadcast_to(np.asarray(f(t), dtype=complex), t.shape))
    return tuple(np.where(np.abs(v.imag) < 1e-12, v.real, np.nan) for v in out)

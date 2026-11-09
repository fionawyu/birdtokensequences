"""Audit the symbolic differentiability of the fitted contour formulas.

Answers three reviewer questions with measurements rather than assertions:

  Q1. What if the formula is not differentiable?
  Q2. Did you differentiate the formula symbolically?
  Q3. Is SymPy guaranteed to differentiate any differentiable formula
      correctly, or does it have limitations?

The script

  * parses every fitted formula (the `fmlfreq` S-expression in the
    *_formulae.yml files) into a SymPy expression,
  * differentiates each piecewise branch once and twice with sympy.diff,
  * checks each symbolic derivative against an independent, hand-coded
    second-order forward-mode AD implementation evaluated at 120 digits,
    at points drawn from the branch's own time support, so "SymPy was
    right" is verified rather than assumed,
  * counts, over the branch's actual support, how often f, f' or f'' is
    undefined or non-finite (the abs/sqrt/log/tan/division cases).

Usage:
    python scripts/sympy_differentiability_audit.py [--root DIR]
                                                    [--limit N]
                                                    [--out results_dir]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import math
import sys
import time
from collections import Counter

import mpmath
import numpy as np
import sympy as sp
import yaml

# t is declared real: this matters. Over a real symbol SymPy differentiates
# Abs(u) to sign(u)*u'; over a default (complex) symbol it produces a
# derivative in terms of re/im and refuses to simplify.
T = sp.Symbol("t", real=True)

UNARY = {
    "sqr": lambda a: a ** 2,
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "neg": lambda a: -a,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "abs": sp.Abs,
}

BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}

RELATION = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}

# Operators whose derivative can be undefined at isolated points even though
# the operator itself has a differentiation rule.
RISKY = {"abs", "sqrt", "log", "tan", "/"}


# --------------------------------------------------------------------------
# S-expression front end
# --------------------------------------------------------------------------
def tokenize(s: str):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def parse(tokens, pos=0):
    """Return (node, next_pos). A node is a str atom or a list."""
    if pos >= len(tokens):
        raise ValueError("unexpected end of formula")
    tok = tokens[pos]
    if tok == "(":
        out = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            child, pos = parse(tokens, pos)
            out.append(child)
        if pos >= len(tokens):
            raise ValueError("unbalanced parentheses")
        return out, pos + 1
    if tok == ")":
        raise ValueError("unexpected )")
    return tok, pos + 1


def parse_formula(s: str):
    node, pos = parse(tokenize(s), 0)
    if pos != len(tokenize(s)):
        raise ValueError("trailing tokens after formula")
    return node


NUM = re.compile(r"^-?(\d+\.?\d*|\.\d+)$")


def to_sympy(node):
    """Convert a parsed node into a SymPy expression (arithmetic subset)."""
    if isinstance(node, str):
        if node == "t":
            return T
        if NUM.match(node):
            # Exact rationals, not Floats: otherwise the AD-vs-SymPy
            # comparison bottoms out at coefficient rounding (~1e-16)
            # instead of showing the true agreement.
            return sp.Rational(node)
        raise ValueError(f"unknown atom {node!r}")
    if not node:
        raise ValueError("empty node")
    head, args = node[0], node[1:]
    if head in UNARY and len(args) == 1:
        return UNARY[head](to_sympy(args[0]))
    if head in BINARY and len(args) == 2:
        return BINARY[head](to_sympy(args[0]), to_sympy(args[1]))
    raise ValueError(f"unsupported node head {head!r}/{len(args)}")


def to_condition(node):
    """Convert a guard node into (sympy_relational, (lo, hi) or None)."""
    if isinstance(node, str):
        raise ValueError(f"atomic guard {node!r}")
    head, args = node[0], node[1:]
    if head == "&&":
        conds, spans = zip(*(to_condition(a) for a in args))
        cond = sp.And(*conds)
        lo = max((s[0] for s in spans if s and s[0] is not None), default=None)
        hi = min((s[1] for s in spans if s and s[1] is not None), default=None)
        return cond, (lo, hi)
    if head == "||":
        conds = [to_condition(a)[0] for a in args]
        return sp.Or(*conds), None
    if head in RELATION and len(args) == 2:
        lhs, rhs = to_sympy(args[0]), to_sympy(args[1])
        cond = RELATION[head](lhs, rhs)
        span = None
        if lhs == T and rhs.is_number:
            v = float(rhs)
            span = (v, None) if head in (">", ">=") else (None, v)
        elif rhs == T and lhs.is_number:
            v = float(lhs)
            span = (None, v) if head in (">", ">=") else (v, None)
        return cond, span
    raise ValueError(f"unsupported guard head {head!r}")


def branches_of(node):
    """Yield (expr, cond, span) for each non-else branch of a `cond` form.

    A bare (non-piecewise) formula is returned as a single unguarded branch.
    """
    if isinstance(node, list) and node and node[0] == "cond":
        for clause in node[1:]:
            if not isinstance(clause, list) or len(clause) != 2:
                raise ValueError("malformed cond clause")
            guard, body = clause
            if guard == "else":
                continue
            cond, span = to_condition(guard)
            yield to_sympy(body), cond, span, body
    else:
        yield to_sympy(node), sp.true, None, node


# --------------------------------------------------------------------------
# Per-branch analysis
# --------------------------------------------------------------------------
def op_histogram(node, counter):
    if not isinstance(node, list) or not node:
        return
    # A cond clause is [guard, body]; its head is a list, not an operator.
    rest = node
    if isinstance(node[0], str):
        counter[node[0]] += 1
        rest = node[1:]
    for child in rest:
        op_histogram(child, counter)


def sample_points(span, n):
    lo, hi = (span if span else (0.0, 1.0))
    if lo is None:
        lo = (hi - 1.0) if hi is not None else 0.0
    if hi is None:
        hi = lo + 1.0
    if not (hi > lo):
        hi = lo + 1e-3
    return np.linspace(lo, hi, n)


def finite_fraction(fn, ts):
    """Fraction of sample points where a lambdified function is real-finite."""
    try:
        with np.errstate(all="ignore"):
            v = np.asarray(fn(ts), dtype=complex)
    except Exception:
        # Not vectorizable (e.g. a constant expression) - fall back per point.
        vals = []
        for t in ts:
            try:
                vals.append(complex(fn(float(t))))
            except Exception:
                vals.append(complex("nan"))
        v = np.asarray(vals, dtype=complex)
    if v.ndim == 0:
        v = np.repeat(v, len(ts))
    good = np.isfinite(v.real) & (np.abs(v.imag) < 1e-9)
    return float(good.mean())


class Jet:
    """Second-order forward-mode AD over mpmath reals: (value, f', f'').

    This is an independent, hand-coded implementation of the chain rule for
    exactly the operators in the grammar. It is the oracle against which the
    SymPy derivatives are checked. Finite differences are deliberately NOT
    used: several fitted branches contain sin(exp(9t)), whose oscillation
    period near t=7 is ~1e-27, so any step-based scheme is meaningless there
    while both AD and symbolic evaluation stay exact.
    """

    __slots__ = ("v", "d", "dd")

    def __init__(self, v, d=0, dd=0):
        self.v, self.d, self.dd = mpmath.mpf(v), mpmath.mpf(d), mpmath.mpf(dd)

    # --- arithmetic ---
    def __add__(self, o):
        return Jet(self.v + o.v, self.d + o.d, self.dd + o.dd)

    def __sub__(self, o):
        return Jet(self.v - o.v, self.d - o.d, self.dd - o.dd)

    def __mul__(self, o):
        return Jet(self.v * o.v,
                   self.d * o.v + self.v * o.d,
                   self.dd * o.v + 2 * self.d * o.d + self.v * o.dd)

    def __truediv__(self, o):
        if o.v == 0:
            raise ZeroDivisionError
        v = self.v / o.v
        d = (self.d - v * o.d) / o.v
        dd = (self.dd - 2 * d * o.d - v * o.dd) / o.v
        return Jet(v, d, dd)

    def chain(self, g, g1, g2):
        """Apply a scalar function with known first/second derivatives."""
        return Jet(g(self.v),
                   g1(self.v) * self.d,
                   g2(self.v) * self.d ** 2 + g1(self.v) * self.dd)


def _jet_unary(name, u: Jet) -> Jet:
    if name == "sqr":
        return u.chain(lambda x: x ** 2, lambda x: 2 * x, lambda x: mpmath.mpf(2))
    if name == "sqrt":
        if u.v <= 0:
            raise ValueError("sqrt domain")
        return u.chain(mpmath.sqrt,
                       lambda x: 1 / (2 * mpmath.sqrt(x)),
                       lambda x: -1 / (4 * x ** mpmath.mpf(1.5)))
    if name == "exp":
        return u.chain(mpmath.exp, mpmath.exp, mpmath.exp)
    if name == "neg":
        return Jet(-u.v, -u.d, -u.dd)
    if name == "sin":
        return u.chain(mpmath.sin, mpmath.cos, lambda x: -mpmath.sin(x))
    if name == "cos":
        return u.chain(mpmath.cos, lambda x: -mpmath.sin(x), lambda x: -mpmath.cos(x))
    if name == "tan":
        return u.chain(mpmath.tan,
                       lambda x: 1 + mpmath.tan(x) ** 2,
                       lambda x: 2 * mpmath.tan(x) * (1 + mpmath.tan(x) ** 2))
    if name == "log":
        if u.v <= 0:
            raise ValueError("log domain")
        return u.chain(mpmath.log, lambda x: 1 / x, lambda x: -1 / x ** 2)
    if name == "abs":
        if u.v == 0:
            raise ValueError("abs kink")
        s = mpmath.mpf(1) if u.v > 0 else mpmath.mpf(-1)
        return Jet(abs(u.v), s * u.d, s * u.dd)
    raise ValueError(f"no AD rule for {name}")


def eval_jet(node, t):
    """Evaluate a parsed S-expression and its two derivatives at t."""
    if isinstance(node, str):
        if node == "t":
            return Jet(t, 1, 0)
        return Jet(mpmath.mpf(node))
    head, args = node[0], node[1:]
    if head in UNARY:
        return _jet_unary(head, eval_jet(args[0], t))
    a, b = eval_jet(args[0], t), eval_jet(args[1], t)
    if head == "+":
        return a + b
    if head == "-":
        return a - b
    if head == "*":
        return a * b
    if head == "/":
        return a / b
    raise ValueError(f"no AD rule for {head}")


def eval_jet_float(node, t):
    """Float-mode second-order AD: (f, f', f'') as plain floats.

    Same chain rule as Jet, but ~1000x faster, so it can scan a whole grid.
    Raises on a genuine domain failure (sqrt/log of a non-positive number,
    Abs exactly at its kink, division by zero) - which is precisely the
    "the derivative does not exist here" case we want to count.
    """
    if isinstance(node, str):
        if node == "t":
            return (t, 1.0, 0.0)
        c = float(node)
        return (c, 0.0, 0.0)
    head, args = node[0], node[1:]
    if head in UNARY:
        v, d, dd = eval_jet_float(args[0], t)
        if head == "sqr":
            return (v * v, 2 * v * d, 2 * d * d + 2 * v * dd)
        if head == "neg":
            return (-v, -d, -dd)
        if head == "exp":
            e = math.exp(v)
            return (e, e * d, e * (d * d + dd))
        if head == "sin":
            sv, cv = math.sin(v), math.cos(v)
            return (sv, cv * d, -sv * d * d + cv * dd)
        if head == "cos":
            sv, cv = math.sin(v), math.cos(v)
            return (cv, -sv * d, -cv * d * d - sv * dd)
        if head == "tan":
            tv = math.tan(v)
            s2 = 1 + tv * tv
            return (tv, s2 * d, 2 * tv * s2 * d * d + s2 * dd)
        if head == "sqrt":
            if v <= 0:
                raise ValueError("sqrt domain")
            r = math.sqrt(v)
            return (r, d / (2 * r), -d * d / (4 * v * r) + dd / (2 * r))
        if head == "log":
            if v <= 0:
                raise ValueError("log domain")
            return (math.log(v), d / v, -d * d / (v * v) + dd / v)
        if head == "abs":
            if v == 0:
                raise ValueError("abs kink")
            sgn = 1.0 if v > 0 else -1.0
            return (abs(v), sgn * d, sgn * dd)
        raise ValueError(head)
    (av, ad, add_), (bv, bd, bdd) = (eval_jet_float(args[0], t),
                                     eval_jet_float(args[1], t))
    if head == "+":
        return (av + bv, ad + bd, add_ + bdd)
    if head == "-":
        return (av - bv, ad - bd, add_ - bdd)
    if head == "*":
        return (av * bv, ad * bv + av * bd, add_ * bv + 2 * ad * bd + av * bdd)
    if head == "/":
        if bv == 0:
            raise ZeroDivisionError
        # v = a/b;  v' = (a' - v b')/b;  v'' = (a'' - 2 v' b' - v b'')/b
        # This form never forms b**2 or b**3, so a huge denominator no longer
        # overflows an otherwise perfectly finite quotient.
        v = av / bv
        d = (ad - v * bd) / bv
        dd = (add_ - 2 * d * bd - v * bdd) / bv
        return (v, d, dd)
    raise ValueError(head)


def eval_value_float(node, t):
    """Value only - no derivatives, so a derivative overflow cannot be
    misreported as the value being undefined."""
    if isinstance(node, str):
        return t if node == "t" else float(node)
    head, args = node[0], node[1:]
    if head in UNARY:
        v = eval_value_float(args[0], t)
        if head == "sqr":
            return v * v
        if head == "neg":
            return -v
        if head == "exp":
            return math.exp(v)
        if head == "sin":
            return math.sin(v)
        if head == "cos":
            return math.cos(v)
        if head == "tan":
            return math.tan(v)
        if head == "sqrt":
            if v < 0:
                raise ValueError("sqrt domain")
            return math.sqrt(v)
        if head == "log":
            if v <= 0:
                raise ValueError("log domain")
            return math.log(v)
        if head == "abs":
            return abs(v)
        raise ValueError(head)
    a = eval_value_float(args[0], t)
    b = eval_value_float(args[1], t)
    if head == "+":
        return a + b
    if head == "-":
        return a - b
    if head == "*":
        return a * b
    if head == "/":
        if b == 0:
            raise ZeroDivisionError
        return a / b
    raise ValueError(head)


def domain_scan(node, span, grid=101):
    """Fraction of the branch support where f, f' and f'' are defined+finite."""
    ok = [0, 0, 0]
    ts = sample_points(span, grid)
    for t in ts:
        try:
            v = eval_value_float(node, float(t))
            if math.isfinite(v):
                ok[0] += 1
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
        try:
            _v, d1, d2 = eval_jet_float(node, float(t))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if math.isfinite(d1):
            ok[1] += 1
        if math.isfinite(d2):
            ok[2] += 1
    n = float(len(ts))
    return ok[0] / n, ok[1] / n, ok[2] / n


def verify_against_ad(node, d1, d2, span, n_check=8, dps=120):
    """Check SymPy's symbolic d1/d2 against independent forward-mode AD.

    Returns (checked, max_rel_err_1, max_rel_err_2). Points where the
    function or its derivative is genuinely undefined are skipped, not
    counted as errors.
    """
    old = mpmath.mp.dps
    mpmath.mp.dps = dps
    try:
        g1 = sp.lambdify(T, d1, "mpmath")
        g2 = sp.lambdify(T, d2, "mpmath")
        ts = sample_points(span, n_check + 2)[1:-1]  # skip exact endpoints
        checked = 0
        e1 = e2 = 0.0
        for tv in ts:
            # Cheap float pre-check first. Where the value already overflows
            # a double, the mpmath path would have to reduce an argument of
            # millions of digits modulo 2*pi and effectively hangs, so skip
            # such points instead of verifying them.
            try:
                pre = eval_jet_float(node, float(tv))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            if not all(math.isfinite(x) and abs(x) < 1e100 for x in pre):
                continue
            tv = mpmath.mpf(float(tv))
            try:
                jet = eval_jet(node, tv)
                sym1, sym2 = g1(tv), g2(tv)
            except Exception:
                continue
            try:
                pairs = ((jet.d, sym1, 1), (jet.dd, sym2, 2))
                errs = []
                for ref, sym, slot in pairs:
                    sym = mpmath.mpf(sym.real if hasattr(sym, "real") else sym)
                    if not (mpmath.isfinite(ref) and mpmath.isfinite(sym)):
                        raise ArithmeticError
                    scale = max(abs(ref), abs(sym), mpmath.mpf(1))
                    errs.append((slot, float(abs(ref - sym) / scale)))
            except (ArithmeticError, TypeError, ValueError):
                continue
            for slot, err in errs:
                if slot == 1:
                    e1 = max(e1, err)
                else:
                    e2 = max(e2, err)
            checked += 1
        return checked, e1, e2
    finally:
        mpmath.mp.dps = old


def risky_ops(node):
    c = Counter()
    op_histogram(node, c)
    return sorted(set(c) & RISKY)


_DIFF_CACHE = {}
_LAMBDA_CACHE = {}
_VERIFIED = {}


def lambdify_cached(expr):
    key = sp.srepr(expr)
    hit = _LAMBDA_CACHE.get(key, False)
    if hit is False:
        try:
            hit = sp.lambdify(T, expr, "numpy")
        except Exception as exc:
            hit = exc
        _LAMBDA_CACHE[key] = hit
    if isinstance(hit, Exception):
        raise hit
    return hit


def differentiate_cached(expr):
    """sympy.diff is the expensive step; the same expression recurs across
    thousands of segments, so memoize on the structural key."""
    key = sp.srepr(expr)
    hit = _DIFF_CACHE.get(key)
    if hit is None:
        hit = _DIFF_CACHE[key] = _differentiate(expr)
    return hit


def _differentiate(expr):
    out = {"diff1_ok": False, "diff2_ok": False, "error": None,
           "d1": None, "d2": None, "seconds": None}
    t0 = time.perf_counter()
    try:
        out["d1"] = sp.diff(expr, T)
        out["diff1_ok"] = True
        out["d2"] = sp.diff(expr, T, 2)
        out["diff2_ok"] = True
    except Exception as exc:  # pragma: no cover - has never triggered
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - t0
    return out


def analyze_branch(expr, span, raw_node, grid=101, verify=True):
    """Differentiate one branch twice and characterize the result."""
    rec = {"risky_ops": risky_ops(raw_node)}
    got = differentiate_cached(expr)
    rec["diff1_ok"] = got["diff1_ok"]
    rec["diff2_ok"] = got["diff2_ok"]
    rec["diff_seconds"] = got["seconds"]
    d1, d2 = got["d1"], got["d2"]
    if not (got["diff1_ok"] and got["diff2_ok"]):
        rec["error"] = got["error"]
        return rec, d1, d2

    # Does the derivative still contain unevaluated Derivative objects, or
    # non-smooth markers (sign/Heaviside/Abs) that SymPy emits at kinks?
    rec["unevaluated"] = bool(d1.atoms(sp.Derivative) or d2.atoms(sp.Derivative))
    rec["has_sign_marker"] = bool(
        d1.atoms(sp.sign) or d2.atoms(sp.sign)
        or d1.atoms(sp.Heaviside) or d2.atoms(sp.Heaviside)
        or d1.atoms(sp.Abs) or d2.atoms(sp.Abs)
    )
    rec["has_dirac"] = bool(d1.atoms(sp.DiracDelta) or d2.atoms(sp.DiracDelta))

    rec["finite_f"], rec["finite_d1"], rec["finite_d2"] = domain_scan(
        raw_node, span, grid)

    # Separately: can SymPy's own symbolic derivative be compiled to a
    # numeric function at all? Cached, because it is the expensive step.
    for key, ex in ((("d1", d1), ("d2", d2)) if verify else ()):
        try:
            lambdify_cached(ex)
        except Exception as exc:
            rec.setdefault("not_evaluable", []).append(
                f"{key}: {type(exc).__name__}")

    if not verify:
        rec["numeric_checked"] = 0
        return rec, d1, d2

    verify_key = sp.srepr(expr)
    if verify_key in _VERIFIED:
        checked, e1, e2 = _VERIFIED[verify_key]
        rec["verify_cached"] = True
    else:
        try:
            checked, e1, e2 = verify_against_ad(raw_node, d1, d2, span)
            _VERIFIED[verify_key] = (checked, e1, e2)
        except Exception as exc:
            checked, e1, e2 = 0, 0.0, 0.0
            rec["verify_error"] = f"{type(exc).__name__}: {exc}"[:200]
    rec["numeric_checked"] = checked
    rec["max_rel_err_d1"] = e1
    rec["max_rel_err_d2"] = e2
    return rec, d1, d2


# --------------------------------------------------------------------------
# Corpus driver
# --------------------------------------------------------------------------
def collect_files(root, limit=None, seed=0):
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith("_formulae.yml"):
                hits.append(os.path.join(dirpath, name))
    hits.sort()
    if limit and len(hits) > limit:
        random.Random(seed).shuffle(hits)
        hits = sorted(hits[:limit])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=None,
                    help="analyze a random sample of N formula files")
    ap.add_argument("--out", default="results_symbolic_differentiation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relative-domain", action="store_true",
                    help="Evaluate each branch on [0, hi-lo] rather than on "
                         "its guard interval [lo, hi]. This is the domain the "
                         "subformula was actually FITTED on: "
                         "main_*_revised.synthesize_frequency passes "
                         "time_seg = times[0:seg[1]-seg[0]], which starts at "
                         "0, while utils.composeFinalFml wraps the branch in "
                         "an absolute-time guard without shifting t.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the high-precision AD cross-check (the "
                         "expensive step); still parses, differentiates and "
                         "scans the domain of every branch")
    args = ap.parse_args()

    files = collect_files(args.root, args.limit, args.seed)
    print(f"[audit] {len(files)} formula files", flush=True)

    ops = Counter()
    stats = Counter()
    rel_err1 = rel_err2 = 0.0
    seen_exprs = {}
    per_expr_rows = []
    failures = []
    nonsmooth_examples = []
    partial_domain = []
    dirac_examples = []
    unevaluable = []
    undefined_value = []

    for i, path in enumerate(files):
        try:
            with open(path) as fh:
                doc = yaml.safe_load(fh)
        except Exception as exc:
            failures.append({"file": path, "stage": "yaml", "error": str(exc)})
            continue
        src = doc.get("fmlfreq")
        if not isinstance(src, str) or not src.strip():
            stats["files_without_formula"] += 1
            continue
        stats["files"] += 1
        try:
            node = parse_formula(src)
            op_histogram(node, ops)
            branches = list(branches_of(node))
        except Exception as exc:
            failures.append({"file": path, "stage": "parse", "error": str(exc)})
            stats["files_parse_failed"] += 1
            continue

        for expr, _cond, span, raw_node in branches:
            stats["branches"] += 1
            key = (sp.srepr(expr), span)
            if key in seen_exprs:
                stats["branches_duplicate"] += 1
                continue
            seen_exprs[key] = True
            stats["branches_unique"] += 1

            scan_span = span
            if args.relative_domain and span and None not in span:
                scan_span = (0.0, span[1] - span[0])
            rec, d1, d2 = analyze_branch(expr, scan_span, raw_node,
                                         verify=not args.no_verify)
            rec["scan_span"] = scan_span
            rec["expr"] = str(expr)
            rec["span"] = span
            rec["file"] = path
            if d1 is not None:
                rec["d1"] = str(d1)
            if d2 is not None:
                rec["d2"] = str(d2)

            if rec.get("diff1_ok") and rec.get("diff2_ok"):
                stats["diff_ok"] += 1
            else:
                stats["diff_failed"] += 1
                failures.append({"file": path, "stage": "diff",
                                 "expr": str(expr),
                                 "error": rec.get("error", "")})
            if rec.get("unevaluated"):
                stats["unevaluated_derivative"] += 1
            if rec.get("has_sign_marker"):
                stats["nonsmooth_marker"] += 1
                if len(nonsmooth_examples) < 25:
                    nonsmooth_examples.append(rec)
            if rec.get("has_dirac"):
                stats["distributional_derivative"] += 1
                if len(dirac_examples) < 25:
                    dirac_examples.append(rec)
            if rec.get("not_evaluable"):
                stats["derivative_not_lambdifiable"] += 1
                if len(unevaluable) < 25:
                    unevaluable.append(rec)
            fd1, fd2 = rec.get("finite_d1"), rec.get("finite_d2")
            if (fd1 is not None and fd1 < 1.0) or (fd2 is not None and fd2 < 1.0):
                stats["derivative_undefined_somewhere"] += 1
                if len(partial_domain) < 25:
                    partial_domain.append(rec)
            ff = rec.get("finite_f")
            if ff is not None and ff < 1.0:
                stats["value_undefined_somewhere"] += 1
                if ff == 0.0:
                    stats["value_undefined_everywhere_on_support"] += 1
                if len(undefined_value) < 25:
                    undefined_value.append(rec)
            if rec.get("numeric_checked", 0) > 0:
                stats["numerically_verified"] += 1
                rel_err1 = max(rel_err1, rec["max_rel_err_d1"])
                rel_err2 = max(rel_err2, rec["max_rel_err_d2"])
            else:
                stats["numeric_check_skipped"] += 1
            per_expr_rows.append(rec)

        if (i + 1) % 200 == 0:
            print(f"[audit] {i + 1}/{len(files)} files, "
                  f"{stats['branches_unique']} unique branches", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stats["unique_expressions_differentiated"] = len(_DIFF_CACHE)
    summary = {
        "sympy_version": sp.__version__,
        "files_scanned": len(files),
        "stats": dict(stats),
        "operator_counts": dict(ops.most_common()),
        "max_rel_err_first_derivative": rel_err1,
        "max_rel_err_second_derivative": rel_err2,
        "parse_or_diff_failures": failures[:50],
        "n_failures": len(failures),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out, "per_branch.json"), "w") as fh:
        json.dump(per_expr_rows, fh, indent=2)
    with open(os.path.join(args.out, "nonsmooth_examples.json"), "w") as fh:
        json.dump({"sign_markers": nonsmooth_examples,
                   "undefined_derivative_on_support": partial_domain,
                   "distributional_derivative": dirac_examples,
                   "not_lambdifiable": unevaluable,
                   "value_undefined_on_support": undefined_value},
                  fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"SymPy {sp.__version__}")
    for k, v in sorted(stats.items()):
        print(f"  {k:34s} {v}")
    print(f"  max rel err  f'  vs mpmath        {rel_err1:.3e}")
    print(f"  max rel err  f'' vs mpmath        {rel_err2:.3e}")
    print("\noperators used:")
    for k, v in ops.most_common():
        print(f"  {k:8s} {v}")
    print(f"\nwrote {args.out}/summary.json, per_branch.json, "
          f"nonsmooth_examples.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

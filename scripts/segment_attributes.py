"""Segment attributes (Section 2.6) from the fitted segment formulas.

For every segment with a fitted formula this computes ten attributes:

    duration_s                              segment duration, real seconds
    start_hz, end_hz, min_hz, max_hz,       frequency attributes, Hz
    mean_hz, range_hz
    slope_hz_per_s                          mean first derivative
    curvature_hz_per_s2                     mean |second derivative|
    linear_dev_hz                           RMSE of a straight-line fit

Each subformula is evaluated where it was fitted. synthesize_frequency fits
segment [s0, s1) against time_seg = times[0:n], n = s1 - s0, on the
stretched axis times = linspace(0, D * times_extended, W, endpoint=False),
and fmlDrawSpec samples it at those same n points t_i = i * step,
step = D * times_extended / W. The attributes use exactly these points.

f, f' and f'' are obtained symbolically with SymPy (sympy.diff) and
evaluated with sympy.lambdify; --verify re-checks them against the
independent second-order AD in scripts/sympy_differentiability_audit.py.

Units. Real time is t / times_extended. The formula value v is matched in
fmlDrawSpec against normed_freqs = linspace(0, uyFreqlim / 12000, H), whose
row i is the cut spectrogram row of frequency freqs[i]; freqs is uniformly
spaced, so the continuous map is affine,
    Hz(v) = freqs[0] + v * (freqs[-1] - freqs[0]) * 12000 / uyFreqlim,
which makes derivatives convert by constant factors:
    slope [Hz/s]       = k * times_extended    * mean f'
    curvature [Hz/s^2] = k * times_extended**2 * mean |f''|,
with k = (freqs[-1] - freqs[0]) * 12000 / uyFreqlim. Values outside the
band are not clipped; `frac_out_of_band` reports how often that happens.

Segment indices, W, D and freqs are recomputed per recording with the
species' own driver (main_*_revised.py: its loader, getSpecSegments_smooth
and parameters), and each recomputed segment is matched to the stored
boundary (start, end) = (times[s0], times[min(s1, W - 1)]) rounded to 3
decimals; `match_error` records the residual. Silence segments (token
ZERO, formula "0") have no fitted formula and get no attributes.

Usage (from the repo root, where the audio is):
    python scripts/segment_attributes.py \
        --segments-csv diagnosis/chapter1_diagnosis/segments.csv \
        --out results_segment_attributes/segment_attributes.csv --verify
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sympy_differentiability_audit import (  # noqa: E402
    T, eval_jet_float, parse_formula, to_sympy,
)

ATTRIBUTES = [
    "duration_s",
    "start_hz", "end_hz", "min_hz", "max_hz", "mean_hz", "range_hz",
    "slope_hz_per_s", "curvature_hz_per_s2", "linear_dev_hz",
]

# dataset -> species driver whose loader, segmentation and parameters
# produced its formulas
DRIVERS = {
    "1017_plots47_Chickadee_1017": "main_chickeadee_revised",
    "1017_plots47_TEWA_1017": "main_TEWA_revised",
    "1017_plots47_WTSP_1017": "main_WTSP_revised",
    "1017_plots47_feebee_1017": "main_feebee_revised",
    "1017_plots47_YWAR_1017": "main_YWAR_revised",
    "1017_plots47_YRWA_1017": "main_YRWA_revised",
    "1017_plots47_Tseet_1017": "main_TSEET_revised",
    "1017_plots47_CoCurate_1017": "main_YNCC_Co_revised",
    "1017_plots47_UCurate_1017": "main_YNCC_U_revised",
    "1017_plots47_RNCurate_1017": "main_YNCC_RN_revised",
    "1017_plots47_SLCurate_1017": "main_YNCC_SL_revised",
}

# the YNCC spectrogram .mat files moved to the archive; same file names
YNCC_MAT_DIRS = {
    "1017_plots47_CoCurate_1017": "1017_plots47_Co_1017",
    "1017_plots47_UCurate_1017": "1017_plots47_U_1017",
    "1017_plots47_RNCurate_1017": "1017_plots47_RN_1017",
    "1017_plots47_SLCurate_1017": "1017_plots47_SL_1017",
}
YNCC_MAT_ROOT = REPO / "datasets_link" / "Plots_YNCC_2s_curate_90"

MATCH_TOL = 2e-3  # boundaries are stored rounded to 3 decimals


# ---------------------------------------------------------------- formulas
_DERIV_CACHE: dict[str, tuple] = {}

# Abs differentiates to sign(u) u' and then to DiracDelta(u) terms. Off the
# kink DiracDelta is 0; at the kink (u == 0) the derivative does not exist,
# so both map to nan there rather than to SymPy's sign(0) = 0.
_NUMERIC = [{
    "DiracDelta": lambda x, *k: np.where(np.asarray(x) == 0, np.nan, 0.0),
    "sign": lambda x: np.where(np.asarray(x) == 0, np.nan, np.sign(x)),
}, "numpy"]


def sympy_derivatives(formula: str):
    """lambdified (f, f', f'') of a subformula, via sympy.diff"""
    hit = _DERIV_CACHE.get(formula)
    if hit is None:
        expr = to_sympy(parse_formula(formula))
        # The grammar is evaluated in real arithmetic, but SymPy cannot always
        # prove an Abs argument real (e.g. Abs(log(u)) or Abs(sqrt(u))) and
        # then differentiates it as a complex modulus, leaving re/im/arg terms
        # it cannot differentiate further. sqrt(u**2) is |u| wherever the
        # formula is defined, has derivative u u'/sqrt(u**2) = sign(u) u',
        # and evaluates to 0/0 = nan at the kink u = 0. For arguments SymPy
        # knows are real it folds straight back to Abs(u).
        expr = expr.replace(sp.Abs, lambda u: sp.sqrt(u ** 2))
        d1 = sp.diff(expr, T)
        d2 = sp.diff(d1, T)
        hit = tuple(sp.lambdify(T, e, _NUMERIC) for e in (expr, d1, d2))
        _DERIV_CACHE[formula] = hit
    return hit


def eval_jet_sympy(formula: str, ts: np.ndarray):
    """f, f', f'' at ts as float arrays; undefined or non-real -> nan"""
    out = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in sympy_derivatives(formula):
            v = np.asarray(fn(ts))
            if np.iscomplexobj(v):
                v = np.where(np.abs(v.imag) < 1e-12, v.real, np.nan)
            out.append(np.broadcast_to(v.astype(float), ts.shape).copy())
    return out


def verify_against_ad(formula: str, ts: np.ndarray, jet) -> float:
    """max relative difference between SymPy and the independent AD; inf if
    they disagree on whether (f, f', f'') is defined at some point"""
    node = parse_formula(formula)
    worst = 0.0
    for i, t in enumerate(ts):
        sym_ok = all(math.isfinite(jet[k][i]) for k in range(3))
        try:
            ad = eval_jet_float(node, float(t))
        except (ValueError, ZeroDivisionError, OverflowError):
            # the AD rejects the whole point (e.g. abs at its kink, where
            # f itself is defined but f' is not)
            if sym_ok:
                return math.inf
            continue
        if sym_ok != all(math.isfinite(x) for x in ad):
            return math.inf
        for k in range(3):
            a, b = ad[k], jet[k][i]
            if not (math.isfinite(a) and math.isfinite(b)):
                if math.isfinite(a) != math.isfinite(b):
                    return math.inf
                continue
            worst = max(worst, abs(a - b) / max(1.0, abs(a)))
    return worst


def attributes(formula, n, step, times_extended, hz0, hz_per_unit):
    """the ten attributes of one segment, plus validity diagnostics"""
    ts = step * np.arange(n)
    f, d1, d2 = eval_jet_sympy(formula, ts)
    valid = bool(np.isfinite(f).all() and np.isfinite(d1).all()
                 and np.isfinite(d2).all())
    out = {"n_frames": int(n), "valid": valid,
           "duration_s": n * step / times_extended}
    if not valid:
        out.update({a: np.nan for a in ATTRIBUTES[1:]})
        return out, ts, (f, d1, d2)
    hz = hz0 + hz_per_unit * f
    t_real = ts / times_extended
    if n >= 2:
        line = np.polyval(np.polyfit(t_real, hz, 1), t_real)
        lin_dev = float(np.sqrt(np.mean((hz - line) ** 2)))
    else:
        lin_dev = 0.0
    out.update(
        start_hz=float(hz[0]), end_hz=float(hz[-1]),
        min_hz=float(hz.min()), max_hz=float(hz.max()),
        mean_hz=float(hz.mean()), range_hz=float(hz.max() - hz.min()),
        slope_hz_per_s=float(hz_per_unit * times_extended * d1.mean()),
        curvature_hz_per_s2=float(hz_per_unit * times_extended ** 2
                                  * np.abs(d2).mean()),
        linear_dev_hz=lin_dev,
    )
    return out, ts, (f, d1, d2)


# -------------------------------------------------------------- recordings
_DRIVER_CACHE: dict[str, object] = {}


def load_driver(name):
    if name not in _DRIVER_CACHE:
        saved = sys.argv
        sys.argv = [name, "47"]  # drivers read the RNG seed from argv[1]
        try:
            _DRIVER_CACHE[name] = importlib.import_module(name)
        finally:
            sys.argv = saved
    return _DRIVER_CACHE[name]


def resolve_path(dataset, file_path):
    p = REPO / file_path
    if p.exists():
        return p
    if dataset in YNCC_MAT_DIRS:
        q = YNCC_MAT_ROOT / YNCC_MAT_DIRS[dataset] / Path(file_path).name
        if q.exists():
            return q
    raise FileNotFoundError(file_path)


def recording_geometry(dataset, path):
    """segments, step, freq conversion and times_extended of one recording,
    recomputed exactly as the species driver did when fitting"""
    m = load_driver(DRIVERS[dataset])
    # the loader each driver's own main() calls: the bird drivers use
    # process_spectrogram (their load_prepare_spec is a different, uncut
    # loader), the YNCC drivers only have the .wav/.mat load_prepare_spec
    loader = getattr(m, "process_spectrogram", None) or m.load_prepare_spec
    rec = loader(path, m.noise_threshold, m.amp_threshold, m.lyFreqlim, m.uyFreqlim)
    spec, freqs, duration = rec["denoised_spec"], np.asarray(rec["freqs"], float), rec["duration"]
    W = spec.shape[1]
    frame_ms = duration / W * 1000.0
    segments, _mask, _a = m.getSpecSegments_smooth(
        spec, ampTol_hi=m.ampTol_hi, ampTol_lo=m.ampTol_lo,
        min_gap_frames=int(round(m.min_gap_ms / frame_ms)),
        min_dur_frames=int(round(m.min_dur_ms / frame_ms)),
        quantile=m.quantile, use_quantile=m.use_quantile)
    times = np.linspace(0, duration * m.times_extended, W, endpoint=False)
    dfreq = np.diff(freqs)
    if not np.allclose(dfreq, dfreq.mean(), rtol=1e-6):
        raise ValueError(f"{path}: frequency axis is not uniform")
    return dict(
        segments=[(int(a), int(b)) for a, b in segments],
        bounds=[(times[a], times[min(b, W - 1)]) for a, b in segments],
        step=duration * m.times_extended / W,
        times_extended=m.times_extended,
        hz0=float(freqs[0]),
        hz_per_unit=float((freqs[-1] - freqs[0]) * 12000.0 / m.uyFreqlim),
        v_max=m.uyFreqlim / 12000.0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments-csv", type=Path,
                    default=REPO / "diagnosis" / "chapter1_diagnosis" / "segments.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_segment_attributes" / "segment_attributes.csv")
    ap.add_argument("--verify", action="store_true",
                    help="re-check SymPy derivatives against the independent AD")
    args = ap.parse_args()

    seg = pd.read_csv(args.segments_csv)
    seg["subformula"] = seg["subformula"].astype(str).str.strip()
    fitted = seg[(seg["token"] != "ZERO") & (seg["subformula"] != "0")]
    print(f"{len(seg)} segments, {len(fitted)} with a fitted formula, "
          f"{fitted.groupby(['dataset', 'song_id']).ngroups} recordings")

    rows, problems = [], []
    for (dataset, file_path), grp in fitted.groupby(["dataset", "file_path"], sort=False):
        try:
            geo = recording_geometry(dataset, resolve_path(dataset, file_path))
        except Exception as exc:  # noqa: BLE001 - report and continue
            problems.append(f"{dataset} {file_path}: {type(exc).__name__}: {exc}")
            continue
        bounds = np.array(geo["bounds"]) if geo["bounds"] else np.zeros((0, 2))
        for _i, r in grp.iterrows():
            base = {k: r[k] for k in ["dataset", "song_id", "segment_id", "token",
                                      "label", "start", "end", "subformula"]}
            if len(bounds) == 0:
                problems.append(f"{dataset} {r['song_id']} seg {r['segment_id']}: no segments recomputed")
                continue
            err = np.abs(bounds - [r["start"], r["end"]]).max(axis=1)
            j = int(err.argmin())
            if err[j] > MATCH_TOL:
                problems.append(f"{dataset} {r['song_id']} seg {r['segment_id']}: "
                                f"no recomputed segment within {MATCH_TOL} (best {err[j]:.4f})")
                continue
            s0, s1 = geo["segments"][j]
            att, ts, jet = attributes(r["subformula"], s1 - s0, geo["step"],
                                      geo["times_extended"], geo["hz0"], geo["hz_per_unit"])
            att["match_error"] = float(err[j])
            att["frac_out_of_band"] = float(np.mean((jet[0] < 0) | (jet[0] > geo["v_max"])))
            if args.verify:
                att["sympy_vs_ad_max_rel"] = verify_against_ad(r["subformula"], ts, jet)
            rows.append({**base, **att})

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} segment rows to {args.out}")
    print(f"  valid (f, f', f'' finite and real at every sample point): "
          f"{int(out['valid'].sum())}/{len(out)}")
    print(f"  segments with any out-of-band sample: {int((out['frac_out_of_band'] > 0).sum())}, "
          f"mean out-of-band fraction {out['frac_out_of_band'].mean():.4f}")
    print(f"  max boundary match error: {out['match_error'].max():.5f}")
    if args.verify:
        v = out["sympy_vs_ad_max_rel"]
        print(f"  SymPy vs AD: max rel diff {v.max():.2e}; segments > 1e-8: {int((v > 1e-8).sum())}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

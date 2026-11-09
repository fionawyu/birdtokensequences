#!/usr/bin/env python3
"""Parameter sensitivity analysis for the formula -> token mapping rule
(sampleClass.py: MONO_THR, SLOPE_FRAC_THR, FLAT_RANGE).

Reviewer concern this addresses: token assignments (WIGGLE/FLAT/UP/DOWN)
depend on these three fixed thresholds -- were they chosen arbitrarily, and
would the classification conclusions change under reasonable variation?

Goal is NOT to find a better threshold. It is to show the main
classification conclusion (Token-method species-clustering accuracy across
the 6 SNR conditions) is stable across reasonable variations of each
threshold, varied ONE AT A TIME (the other two held at their default/
production value) -- not a full factorial grid, matching the reviewer's
suggested protocol.

Reuses the already-fit formula.yml files from the SNR noise-injection full
run (65 recordings x 6 conditions) -- no GP fitting, no embeddings; only the
classify_subfmls_by_segments + edit-distance + k-medoids step is rerun per
grid point, using sampleClass.py's mono_thr/slope_frac_thr/flat_range
override parameters (added for this analysis; defaults reproduce the
unchanged production behavior).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from decompose import decomp_fml_string  # noqa: E402
from sampleClass import classify_subfmls_by_segments, MONO_THR, SLOPE_FRAC_THR, FLAT_RANGE  # noqa: E402
from combined_model.common import kmedoids_fit, clustering_accuracy  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
SNR_EXP_DIR = REPO_ROOT / "snr_noise_injection_experiment"
RESULTS_DIR = Path("outputs/noise_injection_pilot_v2/results/by_task")

CLUSTERING_SEEDS = [0, 1, 2, 3]

# One-at-a-time grid: vary one parameter, hold the other two at their
# production default (marked with *).
GRID = {
    "MONO_THR": [0.80, 0.85, 0.90, 0.95],  # default 0.90
    "SLOPE_FRAC_THR": [0.70, 0.75, 0.80, 0.85, 0.90],  # default 0.80
    "FLAT_RANGE": [0.00005, 0.0001, 0.0002, 0.0005, 0.001],  # default 0.0001
}
DEFAULTS = {"MONO_THR": MONO_THR, "SLOPE_FRAC_THR": SLOPE_FRAC_THR, "FLAT_RANGE": FLAT_RANGE}


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1))
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = 0.0 if a[i - 1] == b[j - 1] else 1.0
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + sub)
    return float(dp[m, n])


def pairwise_edit_distance(sequences):
    n = len(sequences)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = edit_distance(sequences[i], sequences[j])
            dist[i, j] = dist[j, i] = d
    return dist


def find_formula_yml(species, recording_name, condition):
    base = RESULTS_DIR / species / recording_name / condition
    matches = list(base.rglob("formula.yml"))
    return matches[0] if matches else None


def load_decomposed(species, recording_name, condition, _cache={}):
    """decompose_fml_string is expensive-ish and identical across every grid
    point for a given (species, recording, condition) -- cache it once."""
    key = (species, recording_name, condition)
    if key in _cache:
        return _cache[key]
    fp = find_formula_yml(species, recording_name, condition)
    if fp is None:
        _cache[key] = None
        return None
    d = yaml.safe_load(open(fp))
    fmlfreq = d.get("fmlfreq")
    if not fmlfreq:
        _cache[key] = ([], [])
        return ([], [])
    segments, subfmls = decomp_fml_string(fmlfreq)
    _cache[key] = (segments, subfmls)
    return (segments, subfmls)


def token_accuracy(rows, condition, mono_thr, slope_frac_thr, flat_range):
    species_list, seqs = [], []
    for r in rows:
        decomposed = load_decomposed(r["species"], r["recording_name"], condition)
        if decomposed is None:
            continue
        segments, subfmls = decomposed
        if not segments:
            continue
        labels = classify_subfmls_by_segments(subfmls, segments,
                                               mono_thr=mono_thr, slope_frac_thr=slope_frac_thr,
                                               flat_range=flat_range)
        seqs.append(labels)
        species_list.append(r["species"])

    dist = pairwise_edit_distance(seqs)
    label_set = sorted(set(species_list))
    label_to_int = {s: i for i, s in enumerate(label_set)}
    y_true = np.array([label_to_int[s] for s in species_list], dtype=int)
    accs = [clustering_accuracy(y_true, kmedoids_fit(dist, len(label_set), seed)) * 100.0
            for seed in CLUSTERING_SEEDS]
    return float(np.mean(accs)), float(np.std(accs)), len(species_list)


def main():
    manifest = list(csv.DictReader(open(SNR_EXP_DIR / "manifests" / "full_run_manifest.tsv"), delimiter="\t"))
    conditions = sorted(set(r["condition"] for r in manifest), key=lambda c: (c != "baseline", c))

    # (param, value) points to evaluate: the shared default point once, then
    # each parameter's grid values excluding its own default (already covered).
    points = [("default", DEFAULTS["MONO_THR"], DEFAULTS["SLOPE_FRAC_THR"], DEFAULTS["FLAT_RANGE"], "default")]
    for param, values in GRID.items():
        for v in values:
            if v == DEFAULTS[param]:
                continue
            kwargs = dict(DEFAULTS)
            kwargs[param] = v
            points.append((param, kwargs["MONO_THR"], kwargs["SLOPE_FRAC_THR"], kwargs["FLAT_RANGE"], v))

    rows_out = []
    for cond in conditions:
        cond_rows = [r for r in manifest if r["condition"] == cond]
        for param, mono_thr, slope_frac_thr, flat_range, value in points:
            acc_mean, acc_sd, n = token_accuracy(cond_rows, cond, mono_thr, slope_frac_thr, flat_range)
            rows_out.append(dict(condition=cond, varied_param=param, value=value, n=n,
                                  mono_thr=mono_thr, slope_frac_thr=slope_frac_thr, flat_range=flat_range,
                                  token_acc_mean=round(acc_mean, 1), token_acc_sd=round(acc_sd, 1)))
            print(f"{cond:10s} {param:15s}={value!s:8s} n={n:3d} token_acc={acc_mean:5.1f}+/-{acc_sd:4.1f}")

    out_path = EXP_DIR / "results" / "sensitivity_results.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_path}")

    # Summary: per condition, range of token_acc_mean across all grid points
    # vs. the default -- the number a reviewer actually wants to see.
    print("\n=== Summary: token_acc_mean range across all grid points, per condition ===")
    for cond in conditions:
        vals = [r["token_acc_mean"] for r in rows_out if r["condition"] == cond]
        default_val = [r["token_acc_mean"] for r in rows_out if r["condition"] == cond and r["varied_param"] == "default"][0]
        print(f"{cond:10s} default={default_val:5.1f}  range=[{min(vals):5.1f}, {max(vals):5.1f}]  "
              f"max_deviation={max(abs(v - default_val) for v in vals):4.1f} pts")


if __name__ == "__main__":
    main()

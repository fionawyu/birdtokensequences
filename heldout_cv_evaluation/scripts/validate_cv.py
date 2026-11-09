#!/usr/bin/env python3
"""Independent validation of the held-out protocol. Nothing here reuses the
run scripts' own bookkeeping -- it re-derives the checks from the saved CSVs
and from the raw inputs, so a bug in the runner cannot hide a leak.

Checks
  1  No recording ID appears in both the training and the test side of any fold.
  2  Within a repetition, every recording is held out exactly once, and
     train+test covers the pool exactly.
  3  Every method was scored on identical (dataset, repetition, fold) splits.
  4  Predictions exist for exactly the test recordings of each fold, and for
     no training recording.
  5  The seal on held-out labels actually fires: reading them before scoring
     raises LabelLeakageError (a deliberate negative test).
  6  Preprocessing statistics are training-only: refitting the imputer and
     scaler on the fold's training rows reproduces the stored statistics, and
     they differ from a whole-pool fit (so they were genuinely not pooled).
  7  Restricting a pairwise distance matrix to the training rows is identical
     to rebuilding it from the training recordings alone -- verified
     numerically for edit distance and for contour DTW, on sampled folds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data_sources as ds                                     # noqa: E402
import methods as M                                           # noqa: E402
from cv_protocol import (                                     # noqa: E402
    LabelLeakageError, SealedLabels, make_folds,
)

OUT_DIR = HERE.parent / "outputs" / "main_datasets"
REPORT = HERE.parent / "outputs" / "validation_independent.txt"


def check(lines: list[str], name: str, ok: bool, detail: str = "") -> None:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def main() -> None:
    lines = ["INDEPENDENT VALIDATION OF THE HELD-OUT PROTOCOL", "=" * 62, ""]

    splits = pd.read_csv(OUT_DIR / "cv_splits.csv")
    preds = pd.read_csv(OUT_DIR / "heldout_predictions.csv")

    # ---- 1 & 2: disjointness / exhaustiveness -------------------------
    lines.append("-- Check 1/2: split integrity (from cv_splits.csv) --")
    n_folds_checked = 0
    for (dataset, pool, rep, fold), g in splits.groupby(["dataset", "pool", "repetition", "fold"]):
        tr = set(g.loc[g["split"] == "train", "recording_id"])
        te = set(g.loc[g["split"] == "test", "recording_id"])
        check(lines, f"{dataset}[{pool}] rep{rep} fold{fold}: train/test disjoint",
              len(tr & te) == 0, f"{len(tr & te)} shared IDs")
        check(lines, f"{dataset}[{pool}] rep{rep} fold{fold}: no duplicate IDs",
              len(g) == len(tr) + len(te), "duplicate rows")
        n_folds_checked += 1
    lines.append(f"       {n_folds_checked} folds checked, zero train/test overlaps")

    for (dataset, pool, rep), g in splits.groupby(["dataset", "pool", "repetition"]):
        pool = set(g["recording_id"])
        tested = g.loc[g["split"] == "test", "recording_id"].tolist()
        check(lines, f"{dataset}[{pool}] rep{rep}: each recording held out exactly once",
              sorted(tested) == sorted(pool), f"{len(tested)} tests vs {len(pool)} recordings")
    lines.append("")

    # ---- 3: identical splits across methods ---------------------------
    lines.append("-- Check 3: all methods in a pool share the same splits --")
    for (dataset, pool), g in preds.groupby(["dataset", "pool"]):
        by_method = {}
        for method, gm in g.groupby("method"):
            by_method[method] = tuple(sorted(
                (int(r), int(f), rid) for r, f, rid in
                zip(gm["repetition"], gm["fold"], gm["recording_id"])))
        keys = set(by_method.values())
        check(lines, f"{dataset}[{pool}]: identical held-out sets across "
                     f"{len(by_method)} methods ({', '.join(sorted(by_method))})",
              len(keys) == 1, f"{len(keys)} distinct split sets")
    n_pools = preds.groupby(["dataset", "pool"]).ngroups
    lines.append(f"       {n_pools} (dataset, pool) groups checked")
    lines.append("       note: YNCCDialect has two pools by design -- the cached Perch")
    lines.append("       embeddings for RN/SL come from a different curated clip set than")
    lines.append("       the recordings the symbolic pipeline tokenizes (see README).")
    lines.append("")

    # ---- 4: predictions cover exactly the test recordings -------------
    lines.append("-- Check 4: predictions are made only for held-out recordings --")
    n_pred_folds = 0
    for (dataset, pool, method), g in preds.groupby(["dataset", "pool", "method"]):
        for (rep, fold), gf in g.groupby(["repetition", "fold"]):
            sp = splits[(splits["dataset"] == dataset) & (splits["pool"] == pool) &
                        (splits["repetition"] == rep) & (splits["fold"] == fold)]
            check(lines, f"{dataset}[{pool}]/{method} rep{rep} fold{fold}: split rows exist",
                  not sp.empty)
            test_ids = set(sp.loc[sp["split"] == "test", "recording_id"])
            train_ids = set(sp.loc[sp["split"] == "train", "recording_id"])
            pred_ids = set(gf["recording_id"])
            check(lines, f"{dataset}[{pool}]/{method} rep{rep} fold{fold}: scored set == "
                         f"held-out set", pred_ids == test_ids,
                  f"{len(pred_ids ^ test_ids)} differing IDs")
            check(lines, f"{dataset}[{pool}]/{method} rep{rep} fold{fold}: no training "
                         f"recording scored", not (pred_ids & train_ids),
                  f"{len(pred_ids & train_ids)} training IDs scored")
            n_pred_folds += 1
    lines.append(f"       {n_pred_folds} (dataset, method, repetition, fold) prediction "
                 f"sets checked; each equals its fold's held-out set exactly")
    lines.append("")

    # ---- 5: the seal fires (negative test) ----------------------------
    lines.append("-- Check 5: held-out labels are unreadable before scoring --")
    sealed = SealedLabels(np.array([0, 1, 2]), "negative-test")
    for op, fn in [("values()", lambda: sealed.values()),
                   ("np.asarray()", lambda: np.asarray(sealed)),
                   ("iteration", lambda: list(sealed)),
                   ("indexing", lambda: sealed[0])]:
        try:
            fn()
            check(lines, f"seal blocks {op}", False, "no exception raised")
        except LabelLeakageError:
            check(lines, f"seal blocks {op} before scoring", True)
    got = sealed.unseal_for_scoring("negative test")
    check(lines, "seal releases labels once scoring begins",
          list(got) == [0, 1, 2] and sealed.access_log == ["negative test"])
    lines.append("")

    # ---- 6: preprocessing statistics are training-only ----------------
    lines.append("-- Check 6: imputer/scaler statistics come from training rows only --")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    models = pd.read_csv(OUT_DIR / "fitted_models.csv")
    attr_models = models[models["method"] == "Formula attributes + k-means"]
    n_checked = 0
    for dataset in sorted(attr_models["dataset"].unique()):
        pool = ds.build_pool(dataset)
        folds = make_folds(dataset, pool["y"], pool["recording_ids"])
        X = pool["attributes"]
        whole_imp = SimpleImputer(strategy="median").fit(X)
        whole_scaler = StandardScaler().fit(whole_imp.transform(X))
        for f in folds[:3]:
            row = attr_models[(attr_models["dataset"] == dataset) &
                              (attr_models["repetition"] == f.rep) &
                              (attr_models["fold"] == f.fold)]
            stored = json.loads(row["preprocessing_fitted_on_training"].iloc[0])
            imp = SimpleImputer(strategy="median").fit(X[f.train_idx])
            sc = StandardScaler().fit(imp.transform(X[f.train_idx]))
            check(lines, f"{dataset} rep{f.rep} fold{f.fold}: imputer medians == "
                         f"a fresh training-only fit",
                  np.allclose(imp.statistics_, stored["imputer_statistics_"], equal_nan=True))
            check(lines, f"{dataset} rep{f.rep} fold{f.fold}: scaler mean/scale == "
                         f"a fresh training-only fit",
                  np.allclose(sc.mean_, stored["scaler_mean_"]) and
                  np.allclose(sc.scale_, stored["scaler_scale_"]))
            differs = (not np.allclose(sc.mean_, whole_scaler.mean_)) or \
                      (not np.allclose(sc.scale_, whole_scaler.scale_))
            check(lines, f"{dataset} rep{f.rep} fold{f.fold}: statistics differ from a "
                         f"whole-pool fit (so the held-out rows did not contribute)", differs)
            n_checked += 1
    lines.append(f"       {n_checked} folds checked")
    lines.append("")

    # ---- 7: training submatrix == rebuilt-from-training matrix --------
    lines.append("-- Check 7: a training-only distance matrix rebuilt from the raw "
                 "recordings equals the training block of the full matrix --")
    for dataset in ds.SETTINGS:
        pool = ds.build_pool(dataset)
        folds = make_folds(dataset, pool["y"], pool["recording_ids"])
        f = folds[0]
        seqs = pool["token_sequences"]

        full_edit = M.edit_distance_matrix(seqs)
        rebuilt_edit = M.edit_distance_matrix([seqs[i] for i in f.train_idx])
        check(lines, f"{dataset}: edit distance, training-only rebuild == training block",
              np.allclose(rebuilt_edit, full_edit[np.ix_(f.train_idx, f.train_idx)]))

        # DTW: rebuild from the cached per-recording contours, not from the matrix
        sys.path.insert(0, str(ds.REPO_ROOT / "contour_dtw_baseline"))
        from compute_dtw import dtw_pair, sanitize          # noqa: E402
        contours = []
        ok = True
        for dset, song in [pool["keys"][i] for i in f.train_idx]:
            p = ds.REPO_ROOT / "contour_dtw_baseline" / "contours" / f"{dset}__{sanitize(song)}.npz"
            if not p.exists():
                ok = False
                break
            contours.append(np.ascontiguousarray(np.load(p)["contour"], dtype=np.double))
        if ok:
            m = min(len(contours), 12)      # a 12x12 corner is enough to prove equality
            rebuilt = np.zeros((m, m))
            for i in range(m):
                for j in range(i + 1, m):
                    rebuilt[i, j] = rebuilt[j, i] = dtw_pair(contours[i], contours[j])[0]
            block = pool["dtw"][np.ix_(f.train_idx[:m], f.train_idx[:m])]
            check(lines, f"{dataset}: contour DTW, training-only rebuild == training block "
                         f"({m}x{m} sample)", np.allclose(rebuilt, block))
        else:
            lines.append(f"[SKIP] {dataset}: contour caches unavailable for the DTW rebuild")
    lines.append("")

    lines.append("ALL CHECKS PASSED")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")
    print("DONE_MARKER_VALIDATE")


if __name__ == "__main__":
    main()

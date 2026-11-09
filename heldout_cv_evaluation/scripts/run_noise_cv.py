#!/usr/bin/env python3
"""Held-out classification for the controlled Gaussian-noise conditions
(manuscript Table tab:controlled_snr): the n=65 BCCH/TEWA/WTSP/YWAR pool at
baseline plus five degraded SNR conditions.

Reuses exactly the inputs the published noise numbers were computed from --
the already-fitted formula.yml files under
outputs/noise_injection_pilot_v2/results/by_task (NOT refitted), the
cached Perch embeddings at
outputs/noise_injection_full/perch_embeddings_full.npz, and the
per-condition contour caches under
manuscript_rerun_more_seeds/contours_target2_noise/ -- and replaces the
whole-dataset clustering with the held-out protocol in cv_protocol.py.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "contour_dtw_baseline"))

import methods as M                                            # noqa: E402
from cv_protocol import (                                      # noqa: E402
    SHUFFLE_SEED_BASE, SealedLabels, assert_identical_splits,
    hungarian_map_from_training, make_folds, summarize_repetitions,
    validate_folds, write_seeds_manifest,
)
from decompose import decomp_fml_string                        # noqa: E402
from sampleClass import classify_subfmls_by_segments           # noqa: E402
from combined_model.attribute_model import (                   # noqa: E402
    compute_segment_features, aggregate_recording_features, NEW_FEATURES,
)
from compute_dtw import dtw_pair, sanitize                     # noqa: E402

OUT_DIR = HERE.parent / "outputs" / "noise_conditions"
MANIFEST = REPO_ROOT / "snr_noise_injection_experiment" / "manifests" / "full_run_manifest.tsv"
FORMULA_DIR = Path("outputs/noise_injection_pilot_v2/results/by_task")
PERCH_CACHE = Path("outputs/noise_injection_full/perch_embeddings_full.npz")
CONTOURS_DIR = REPO_ROOT / "manuscript_rerun_more_seeds" / "contours_target2_noise"

FEATURE_COLS = [f"{f}_{s}" for f in NEW_FEATURES for s in ("mean", "sd", "min", "max")]
SPECIES_TO_DATASET = {
    "BCCH": "1017_plots47_Chickadee_1017", "TEWA": "1017_plots47_TEWA_1017",
    "WTSP": "1017_plots47_WTSP_1017", "YWAR": "1017_plots47_YWAR_1017",
}
CONDITION_LABEL = {
    "baseline": "Baseline", "snr_12db": "12-15 dB", "snr_9db": "9-12 dB",
    "snr_6db": "6-9 dB", "snr_3db": "3-6 dB", "snr_0db": "<=2 dB",
}
CONDITION_ORDER = ["baseline", "snr_12db", "snr_9db", "snr_6db", "snr_3db", "snr_0db"]
METHOD_ORDER = [
    "Token + k-medoids", "Shuffled + k-medoids", "Trajectory-DTW + k-medoids",
    "Formula attributes + k-means", "Perch v2 + k-medoids", "Perch v2 + k-means",
]


def load_formula_tokens(species: str, recording: str, condition: str):
    base = FORMULA_DIR / species / recording / condition
    matches = list(base.rglob("formula.yml"))
    if not matches:
        return None, None
    fmlfreq = (yaml.safe_load(open(matches[0])) or {}).get("fmlfreq")
    if not fmlfreq:
        return [], []
    try:
        segments, subfmls = decomp_fml_string(fmlfreq)
    except Exception:
        return None, None
    if not segments:
        return [], []
    labels = classify_subfmls_by_segments(subfmls, segments)
    seg_rows = [
        {"dataset": species, "song_id": recording, "segment_id": i, "token": lab,
         "subformula": sub,
         "duration": (t2 - t1) if (t1 is not None and t2 is not None) else np.nan}
        for i, ((t1, t2), sub, lab) in enumerate(zip(segments, subfmls, labels))
    ]
    return labels, seg_rows


def build_condition_pool(condition: str, rows: list[dict], perch_cache: dict) -> dict | None:
    species_list, token_seqs, seg_rows_all, wav_paths, rec_names = [], [], [], [], []
    for r in rows:
        seq, seg_rows = load_formula_tokens(r["species"], r["recording_name"], condition)
        if seq is None or not seq:
            continue
        if r["wav_path"] not in perch_cache:
            continue
        cpath = CONTOURS_DIR / condition / f"{SPECIES_TO_DATASET[r['species']]}__{sanitize(r['recording_name'])}.npz"
        if not cpath.exists():
            continue
        species_list.append(r["species"])
        token_seqs.append(seq)
        seg_rows_all.extend(seg_rows)
        wav_paths.append(r["wav_path"])
        rec_names.append(r["recording_name"])
    if not species_list:
        return None

    label_set = sorted(set(species_list))
    label_to_int = {s: i for i, s in enumerate(label_set)}
    y = np.array([label_to_int[s] for s in species_list], dtype=int)

    rec_feats = aggregate_recording_features(compute_segment_features(pd.DataFrame(seg_rows_all)))
    rec_feats = rec_feats.set_index(["dataset", "song_id"]).loc[
        list(zip(species_list, rec_names))].reset_index()

    contours = []
    for sp, name in zip(species_list, rec_names):
        p = CONTOURS_DIR / condition / f"{SPECIES_TO_DATASET[sp]}__{sanitize(name)}.npz"
        contours.append(np.ascontiguousarray(np.load(p)["contour"], dtype=np.double))
    n = len(contours)
    dtw = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dtw[i, j] = dtw[j, i] = dtw_pair(contours[i], contours[j])[0]

    return {
        "condition": condition,
        "recording_ids": [f"{s}||{r}" for s, r in zip(species_list, rec_names)],
        "y": y, "label_set": label_set, "n_classes": len(label_set),
        "token_sequences": token_seqs,
        "attributes": rec_feats[FEATURE_COLS].to_numpy(dtype=float),
        "dtw": dtw,
        "perch": np.stack([perch_cache[p] for p in wav_paths]),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["HELD-OUT CV VALIDATION REPORT -- controlled-noise conditions", "=" * 62, ""]

    manifest = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    with np.load(PERCH_CACHE, allow_pickle=True) as z:
        perch_cache = {k: np.asarray(z[k], dtype=float) for k in z.files}
    print(f"[perch] {len(perch_cache)} cached embeddings", flush=True)

    all_pred, all_map, all_split, all_model = [], [], [], []
    achieved = {}

    for condition in CONDITION_ORDER:
        rows = [r for r in manifest if r["condition"] == condition]
        dbs = [float(r["achieved_db"]) for r in rows if r["achieved_db"]]
        achieved[condition] = float(np.mean(dbs)) if dbs else float("nan")
        pool = build_condition_pool(condition, rows, perch_cache)
        if pool is None:
            print(f"SKIP {condition}: no usable recordings", flush=True)
            continue
        n, k = len(pool["recording_ids"]), pool["n_classes"]
        rec_ids, y = pool["recording_ids"], pool["y"]
        print(f"\n=== {condition} ({CONDITION_LABEL[condition]}): n={n} k={k} "
              f"mean achieved SNR={achieved[condition]:.2f} dB ===", flush=True)

        folds = make_folds(condition, y, rec_ids)
        report.append(f"[{condition}] n={n} classes={k} folds/rep={folds[0].n_splits}")
        report.extend(validate_folds(folds, rec_ids))
        report.extend(assert_identical_splits({m: folds for m in METHOD_ORDER}))

        D_token = M.edit_distance_matrix(pool["token_sequences"])
        D_shuf = {rep: M.edit_distance_matrix(
            M.shuffle_sequences(pool["token_sequences"], rep, SHUFFLE_SEED_BASE))
            for rep in sorted({f.rep for f in folds})}

        for f in folds:
            for split, ids in (("train", f.train_ids), ("test", f.test_ids)):
                for rid in ids:
                    all_split.append({"condition": condition, "repetition": f.rep,
                                      "fold": f.fold, "recording_id": rid, "split": split})
            y_train = y[f.train_idx]
            bs = f.clustering_seed
            runs = {
                "Token + k-medoids": M.run_kmedoids_from_full_matrix(
                    D_token, f.train_idx, f.test_idx, k, bs, rec_ids),
                "Shuffled + k-medoids": M.run_kmedoids_from_full_matrix(
                    D_shuf[f.rep], f.train_idx, f.test_idx, k, bs, rec_ids),
                "Trajectory-DTW + k-medoids": M.run_kmedoids_from_full_matrix(
                    pool["dtw"], f.train_idx, f.test_idx, k, bs, rec_ids),
                "Formula attributes + k-means": M.run_kmeans(
                    pool["attributes"], f.train_idx, f.test_idx, k, bs, use_scaler=True),
                "Perch v2 + k-medoids": M.run_kmedoids_from_vectors(
                    pool["perch"], f.train_idx, f.test_idx, k, bs, rec_ids),
                "Perch v2 + k-means": M.run_kmeans(
                    pool["perch"], f.train_idx, f.test_idx, k, bs, use_scaler=False),
            }
            for method, res in runs.items():
                mapping = hungarian_map_from_training(res["train_clusters"], y_train, k, k)
                pred = np.array([mapping[int(c)] for c in res["test_clusters"]], dtype=int)
                sealed = SealedLabels(y[f.test_idx],
                                      f"{condition}/{method}/rep{f.rep}/fold{f.fold}")
                y_test = sealed.unseal_for_scoring("final scoring; predictions already computed")
                correct = (pred == y_test).astype(int)
                for cluster, cls in sorted(mapping.items()):
                    all_map.append({"condition": condition, "method": method,
                                    "repetition": f.rep, "fold": f.fold, "cluster": cluster,
                                    "mapped_class": pool["label_set"][cls] if cls >= 0 else "UNMAPPED",
                                    "n_training_recordings_in_cluster":
                                        int(np.sum(res["train_clusters"] == cluster))})
                model = dict(res["model"])
                model.pop("centroids", None)
                all_model.append({"condition": condition, "method": method,
                                  "repetition": f.rep, "fold": f.fold,
                                  **{key: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                     for key, v in model.items()}})
                for i, ti in enumerate(f.test_idx):
                    all_pred.append({
                        "condition": condition, "manuscript_label": CONDITION_LABEL[condition],
                        "mean_achieved_db": round(achieved[condition], 2),
                        "method": method, "repetition": f.rep, "fold": f.fold,
                        "split_seed": f.split_seed, "recording_id": rec_ids[ti],
                        "true_class": pool["label_set"][y_test[i]],
                        "predicted_class": pool["label_set"][pred[i]] if pred[i] >= 0 else "UNMAPPED",
                        "assigned_cluster": int(res["test_clusters"][i]),
                        "distance_to_assigned": float(res["test_distance"][i]),
                        "distances_to_all_clusters": json.dumps(
                            [round(float(v), 6) for v in res["all_distances"][i]]),
                        "correct": int(correct[i]),
                    })

    pred_df = pd.DataFrame(all_pred)
    rep_rows, sum_rows = [], []
    for (cond, method), g in pred_df.groupby(["condition", "method"], sort=False):
        accs = []
        for rep, gr in g.groupby("repetition", sort=True):
            acc = 100.0 * gr["correct"].mean()
            accs.append(acc)
            rep_rows.append({"condition": cond, "method": method, "repetition": rep,
                             "n_heldout": len(gr), "accuracy": acc})
        sum_rows.append({"condition": cond, "manuscript_label": CONDITION_LABEL[cond],
                         "mean_achieved_db": round(achieved[cond], 2), "method": method,
                         "n": len(g) // 4, "n_classes": int(g["true_class"].nunique()),
                         **summarize_repetitions(accs)})
    sum_df = pd.DataFrame(sum_rows)
    corder = {c: i for i, c in enumerate(CONDITION_ORDER)}
    morder = {m: i for i, m in enumerate(METHOD_ORDER)}
    sum_df = sum_df.assign(_c=sum_df["condition"].map(corder), _m=sum_df["method"].map(morder))
    sum_df = sum_df.sort_values(["_c", "_m"]).drop(columns=["_c", "_m"])

    pred_df.to_csv(OUT_DIR / "heldout_predictions.csv", index=False)
    pd.DataFrame(all_split).to_csv(OUT_DIR / "cv_splits.csv", index=False)
    pd.DataFrame(all_map).to_csv(OUT_DIR / "cluster_to_class_mapping.csv", index=False)
    pd.DataFrame(all_model).to_csv(OUT_DIR / "fitted_models.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT_DIR / "repetition_accuracy.csv", index=False)
    sum_df.to_csv(OUT_DIR / "heldout_accuracy_summary.csv", index=False)
    write_seeds_manifest(OUT_DIR, {"script": "run_noise_cv.py"})
    (OUT_DIR / "validation_report.txt").write_text("\n".join(report) + "\n")

    print("\n" + sum_df.to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}  [{time.time() - t0:.1f}s]")
    print("DONE_MARKER_HELDOUT_NOISE")


if __name__ == "__main__":
    main()

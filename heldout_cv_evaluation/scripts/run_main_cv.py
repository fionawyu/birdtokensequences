#!/usr/bin/env python3
"""Held-out classification for the five main evaluation datasets.

4 repetitions x stratified 5-fold CV over recording IDs (fewer folds only if a
class had <5 recordings -- none does here). Every method is fitted on the
training recordings of a fold, mapped to classes with the Hungarian algorithm
on the TRAINING labels, and then used to predict the held-out recordings.
Existing results are not touched; everything is written under
heldout_cv_evaluation/outputs/main_datasets/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data_sources as ds                                    # noqa: E402
import methods as M                                          # noqa: E402
from cv_protocol import (                                    # noqa: E402
    SHUFFLE_SEED_BASE, SealedLabels, assert_identical_splits,
    hungarian_map_from_training, make_folds, summarize_repetitions,
    validate_folds, write_seeds_manifest,
)

OUT_DIR = HERE.parent / "outputs" / "main_datasets"

METHOD_ORDER = [
    "Token + k-medoids",
    "Shuffled + k-medoids",
    "Trajectory-DTW + k-medoids",
    "Formula attributes + k-means",
    "Perch v2 + k-medoids",
    "Perch v2 + k-means",
]


def run_setting(setting: str, report: list[str]) -> dict:
    pool = ds.build_pool(setting)
    n = len(pool["keys"])
    k = pool["n_classes"]
    rec_ids = pool["recording_ids"]
    y = pool["y"]

    print(f"\n=== {setting}: n={n}, classes={k} "
          f"({dict(zip(*np.unique([d for d, _ in pool['keys']], return_counts=True)))}) ===",
          flush=True)

    folds = make_folds(setting, y, rec_ids)
    report.append(f"[{setting}] n={n} classes={k} folds/rep={folds[0].n_splits}")
    report.extend(validate_folds(folds, rec_ids))

    checked = M.verify_edit_distance(pool["token_sequences"], n_pairs=300, seed=1)
    report.append(f"  edit distance agrees with the published implementation on "
                  f"{checked} sampled pairs  [OK]")

    # Pairwise matrices over the whole pool; only training blocks are ever fitted on.
    D_token = M.edit_distance_matrix(pool["token_sequences"])
    D_dtw = pool["dtw"]
    D_shuffled_by_rep = {
        rep: M.edit_distance_matrix(
            M.shuffle_sequences(pool["token_sequences"], rep, SHUFFLE_SEED_BASE))
        for rep in sorted({f.rep for f in folds})
    }
    shuffled_seq_by_rep = {
        rep: M.shuffle_sequences(pool["token_sequences"], rep, SHUFFLE_SEED_BASE)
        for rep in sorted({f.rep for f in folds})
    }

    predictions: list[dict] = []
    mappings: list[dict] = []
    models: list[dict] = []
    split_rows: list[dict] = []
    fold_sets: dict[str, list] = {}

    for m in METHOD_ORDER:
        if m.startswith("Perch") and not pool["perch_covers_pool"]:
            continue
        fold_sets[m] = folds  # identical object -> identical splits by construction

    for f in folds:
        for rid in f.train_ids:
            split_rows.append({"dataset": setting, "repetition": f.rep, "fold": f.fold,
                               "pool": "primary", "recording_id": rid, "split": "train"})
        for rid in f.test_ids:
            split_rows.append({"dataset": setting, "repetition": f.rep, "fold": f.fold,
                               "pool": "primary", "recording_id": rid, "split": "test"})

        y_train = y[f.train_idx]
        base_seed = f.clustering_seed

        runs: dict[str, dict] = {}
        runs["Token + k-medoids"] = M.run_kmedoids_from_full_matrix(
            D_token, f.train_idx, f.test_idx, k, base_seed, rec_ids)
        runs["Shuffled + k-medoids"] = M.run_kmedoids_from_full_matrix(
            D_shuffled_by_rep[f.rep], f.train_idx, f.test_idx, k, base_seed, rec_ids)
        runs["Trajectory-DTW + k-medoids"] = M.run_kmedoids_from_full_matrix(
            D_dtw, f.train_idx, f.test_idx, k, base_seed, rec_ids)
        runs["Formula attributes + k-means"] = M.run_kmeans(
            pool["attributes"], f.train_idx, f.test_idx, k, base_seed, use_scaler=True)
        if pool["perch_covers_pool"]:
            runs["Perch v2 + k-medoids"] = M.run_kmedoids_from_vectors(
                pool["perch"], f.train_idx, f.test_idx, k, base_seed, rec_ids)
            runs["Perch v2 + k-means"] = M.run_kmeans(
                pool["perch"], f.train_idx, f.test_idx, k, base_seed, use_scaler=False)

        for method, res in runs.items():
            # --- cluster -> class, from TRAINING labels only -------------
            mapping = hungarian_map_from_training(res["train_clusters"], y_train, k, k)
            pred_class = np.array([mapping[int(c)] for c in res["test_clusters"]], dtype=int)

            # --- only now are the held-out labels unsealed ---------------
            sealed = SealedLabels(y[f.test_idx], f"{setting}/{method}/rep{f.rep}/fold{f.fold}")
            y_test = sealed.unseal_for_scoring("final scoring; predictions already computed")
            correct = (pred_class == y_test).astype(int)

            for cluster, cls in sorted(mapping.items()):
                mappings.append({
                    "dataset": setting, "method": method, "repetition": f.rep, "fold": f.fold,
                    "cluster": cluster,
                    "mapped_class_index": cls,
                    "mapped_class": pool["label_set"][cls] if cls >= 0 else "UNMAPPED",
                    "mapped_class_short": pool["class_names"][cls] if cls >= 0 else "UNMAPPED",
                    "n_training_recordings_in_cluster": int(np.sum(res["train_clusters"] == cluster)),
                })

            model = dict(res["model"])
            centroids = model.pop("centroids", None)
            models.append({
                "dataset": setting, "method": method, "repetition": f.rep, "fold": f.fold,
                "n_train": len(f.train_idx), "n_test": len(f.test_idx),
                **{key: (json.dumps(val) if isinstance(val, (list, dict)) else val)
                   for key, val in model.items()},
                "centroid_shape": None if centroids is None else json.dumps(list(centroids.shape)),
            })

            for i, ti in enumerate(f.test_idx):
                predictions.append({
                    "dataset": setting, "method": method, "pool": "primary",
                    "repetition": f.rep, "fold": f.fold, "split_seed": f.split_seed,
                    "recording_id": rec_ids[ti],
                    "true_class": pool["label_set"][y_test[i]],
                    "true_class_short": pool["class_names"][y_test[i]],
                    "predicted_class": pool["label_set"][pred_class[i]] if pred_class[i] >= 0 else "UNMAPPED",
                    "predicted_class_short": pool["class_names"][pred_class[i]] if pred_class[i] >= 0 else "UNMAPPED",
                    "assigned_cluster": int(res["test_clusters"][i]),
                    "distance_to_assigned": float(res["test_distance"][i]),
                    "distances_to_all_clusters": json.dumps(
                        [round(float(v), 6) for v in res["all_distances"][i]]),
                    "correct": int(correct[i]),
                })

    report.extend(assert_identical_splits(fold_sets))
    return {
        "setting": setting, "pool": pool, "folds": folds,
        "predictions": predictions, "mappings": mappings,
        "models": models, "splits": split_rows,
    }


def run_perch_fallback(setting: str, report: list[str]) -> dict | None:
    """YNCCDialect only: Perch's cached embeddings cover a different recording
    set, so it is evaluated under the identical protocol on its own pool and
    reported with its own n."""
    pool = ds.build_perch_only_pool(setting)
    if pool is None:
        return None
    rec_ids, y, k = pool["recording_ids"], pool["y"], pool["n_classes"]
    folds = make_folds(setting + "(Perch pool)", y, rec_ids)
    report.append(f"[{setting}] Perch-only pool n={len(rec_ids)} classes={k} "
                  f"(different RN/SL recordings -- see README)")
    report.extend(validate_folds(folds, rec_ids))

    predictions, mappings, models, split_rows = [], [], [], []
    for f in folds:
        for rid in f.train_ids:
            split_rows.append({"dataset": setting, "repetition": f.rep, "fold": f.fold,
                               "pool": "perch_only", "recording_id": rid, "split": "train"})
        for rid in f.test_ids:
            split_rows.append({"dataset": setting, "repetition": f.rep, "fold": f.fold,
                               "pool": "perch_only", "recording_id": rid, "split": "test"})
        y_train = y[f.train_idx]
        runs = {
            "Perch v2 + k-medoids": M.run_kmedoids_from_vectors(
                pool["perch"], f.train_idx, f.test_idx, k, f.clustering_seed, rec_ids),
            "Perch v2 + k-means": M.run_kmeans(
                pool["perch"], f.train_idx, f.test_idx, k, f.clustering_seed, use_scaler=False),
        }
        for method, res in runs.items():
            mapping = hungarian_map_from_training(res["train_clusters"], y_train, k, k)
            pred_class = np.array([mapping[int(c)] for c in res["test_clusters"]], dtype=int)
            sealed = SealedLabels(y[f.test_idx], f"{setting}-perch/{method}/rep{f.rep}/fold{f.fold}")
            y_test = sealed.unseal_for_scoring("final scoring; predictions already computed")
            correct = (pred_class == y_test).astype(int)
            for cluster, cls in sorted(mapping.items()):
                mappings.append({
                    "dataset": setting, "method": method, "repetition": f.rep, "fold": f.fold,
                    "cluster": cluster, "mapped_class_index": cls,
                    "mapped_class": pool["label_set"][cls] if cls >= 0 else "UNMAPPED",
                    "mapped_class_short": pool["class_names"][cls] if cls >= 0 else "UNMAPPED",
                    "n_training_recordings_in_cluster": int(np.sum(res["train_clusters"] == cluster)),
                })
            model = dict(res["model"])
            centroids = model.pop("centroids", None)
            models.append({
                "dataset": setting, "method": method, "repetition": f.rep, "fold": f.fold,
                "n_train": len(f.train_idx), "n_test": len(f.test_idx),
                **{key: (json.dumps(val) if isinstance(val, (list, dict)) else val)
                   for key, val in model.items()},
                "centroid_shape": None if centroids is None else json.dumps(list(centroids.shape)),
            })
            for i, ti in enumerate(f.test_idx):
                predictions.append({
                    "dataset": setting, "method": method, "pool": "perch_only",
                    "repetition": f.rep, "fold": f.fold, "split_seed": f.split_seed,
                    "recording_id": rec_ids[ti],
                    "true_class": pool["label_set"][y_test[i]],
                    "true_class_short": pool["class_names"][y_test[i]],
                    "predicted_class": pool["label_set"][pred_class[i]] if pred_class[i] >= 0 else "UNMAPPED",
                    "predicted_class_short": pool["class_names"][pred_class[i]] if pred_class[i] >= 0 else "UNMAPPED",
                    "assigned_cluster": int(res["test_clusters"][i]),
                    "distance_to_assigned": float(res["test_distance"][i]),
                    "distances_to_all_clusters": json.dumps(
                        [round(float(v), 6) for v in res["all_distances"][i]]),
                    "correct": int(correct[i]),
                })
    return {"predictions": predictions, "mappings": mappings,
            "models": models, "splits": split_rows,
            "n": len(rec_ids), "k": k}


def aggregate(pred_df: pd.DataFrame, pool_sizes: dict[tuple[str, str], int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows, rep_rows, sum_rows = [], [], []
    for (dataset, method), g in pred_df.groupby(["dataset", "method"], sort=False):
        for (rep, fold), gf in g.groupby(["repetition", "fold"], sort=True):
            fold_rows.append({"dataset": dataset, "method": method, "repetition": rep,
                              "fold": fold, "n_test": len(gf),
                              "n_correct": int(gf["correct"].sum()),
                              "accuracy": 100.0 * gf["correct"].mean()})
        rep_acc = []
        for rep, gr in g.groupby("repetition", sort=True):
            acc = 100.0 * gr["correct"].mean()
            rep_acc.append(acc)
            rep_rows.append({"dataset": dataset, "method": method, "repetition": rep,
                             "n_heldout": len(gr), "n_correct": int(gr["correct"].sum()),
                             "accuracy": acc})
        n, k = pool_sizes[(dataset, method)]
        sum_rows.append({"dataset": dataset, "method": method, "n": n, "n_classes": k,
                         "chance_accuracy": 100.0 / k, **summarize_repetitions(rep_acc)})
    return pd.DataFrame(fold_rows), pd.DataFrame(rep_rows), pd.DataFrame(sum_rows)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["HELD-OUT CV VALIDATION REPORT -- main datasets", "=" * 62, ""]

    all_pred, all_map, all_model, all_split = [], [], [], []
    pool_sizes: dict[tuple[str, str], tuple[int, int]] = {}

    for setting in ds.SETTINGS:
        out = run_setting(setting, report)
        all_pred += out["predictions"]
        all_map += out["mappings"]
        all_model += out["models"]
        all_split += out["splits"]
        n, k = len(out["pool"]["keys"]), out["pool"]["n_classes"]
        for m in METHOD_ORDER:
            pool_sizes[(setting, m)] = (n, k)

        if not out["pool"]["perch_covers_pool"]:
            fb = run_perch_fallback(setting, report)
            if fb:
                all_pred += fb["predictions"]
                all_map += fb["mappings"]
                all_model += fb["models"]
                all_split += fb["splits"]
                for m in ("Perch v2 + k-medoids", "Perch v2 + k-means"):
                    pool_sizes[(setting, m)] = (fb["n"], fb["k"])

    pred_df = pd.DataFrame(all_pred)
    fold_df, rep_df, sum_df = aggregate(pred_df, pool_sizes)

    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    sum_df = sum_df.assign(_o=sum_df["method"].map(order)).sort_values(
        ["dataset", "_o"]).drop(columns="_o")

    pred_df.to_csv(OUT_DIR / "heldout_predictions.csv", index=False)
    pd.DataFrame(all_split).to_csv(OUT_DIR / "cv_splits.csv", index=False)
    pd.DataFrame(all_map).to_csv(OUT_DIR / "cluster_to_class_mapping.csv", index=False)
    pd.DataFrame(all_model).to_csv(OUT_DIR / "fitted_models.csv", index=False)
    fold_df.to_csv(OUT_DIR / "fold_accuracy.csv", index=False)
    rep_df.to_csv(OUT_DIR / "repetition_accuracy.csv", index=False)
    sum_df.to_csv(OUT_DIR / "heldout_accuracy_summary.csv", index=False)
    write_seeds_manifest(OUT_DIR, {"script": "run_main_cv.py"})

    report.append("")
    report.append(f"Wrote {len(pred_df)} held-out predictions across "
                  f"{pred_df['dataset'].nunique()} datasets and "
                  f"{pred_df['method'].nunique()} methods.")
    (OUT_DIR / "validation_report.txt").write_text("\n".join(report) + "\n")

    print("\n" + sum_df.to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}  [{time.time() - t0:.1f}s]")
    print("DONE_MARKER_HELDOUT_MAIN")


if __name__ == "__main__":
    main()

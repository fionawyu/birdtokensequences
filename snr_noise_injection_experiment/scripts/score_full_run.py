#!/usr/bin/env python3
"""Score the full 65-recording x 6-condition run. Identical methodology to
score_pilot_v2_full.py (Token, Attributes, Perch k-medoids+k-means,
Shuffled-vs-Token), pointed at the full-run manifest and the shared results
dir that also holds the 144 fits reused from pilot v2.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from decompose import decomp_fml_string  # noqa: E402
from sampleClass import classify_subfmls_by_segments  # noqa: E402
from combined_model.common import kmedoids_fit, clustering_accuracy  # noqa: E402
from combined_model.attribute_model import (  # noqa: E402
    compute_segment_features, aggregate_recording_features,
    attribute_distance_matrix, NEW_FEATURES,
)
from classiifier import embed_recordings, kmedoids as perch_kmedoids, clustering_accuracy as perch_clustering_accuracy  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.preprocessing import normalize as l2_normalize  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path("outputs/noise_injection_pilot_v2/results/by_task")
PERCH_CACHE = Path("outputs/noise_injection_full/perch_embeddings_full.npz")

CLUSTERING_SEEDS = [0, 1, 2, 3]
N_SHUFFLES = 20
SHUFFLE_SEED_BASE = 1000
MAJORITY_BASELINE_PCT = 21 / 65 * 100.0


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


def load_formula_data(species, recording_name, condition):
    fp = find_formula_yml(species, recording_name, condition)
    if fp is None:
        return None, None
    d = yaml.safe_load(open(fp))
    fmlfreq = d.get("fmlfreq")
    if not fmlfreq:
        return [], []
    try:
        segments, subfmls = decomp_fml_string(fmlfreq)
    except Exception as e:
        print(f"  [warn] decompose failed for {fp}: {e}", file=sys.stderr)
        return None, None
    if not segments:
        return [], []
    labels = classify_subfmls_by_segments(subfmls, segments)
    seg_rows = [{"dataset": species, "song_id": recording_name, "segment_id": i,
                 "token": lab, "subformula": sub, "duration": (t2 - t1) if (t1 is not None and t2 is not None) else np.nan}
                for i, ((t1, t2), sub, lab) in enumerate(zip(segments, subfmls, labels))]
    return labels, seg_rows


def cluster_and_score(dist, species_list, seeds=CLUSTERING_SEEDS):
    label_set = sorted(set(species_list))
    label_to_int = {s: i for i, s in enumerate(label_set)}
    y_true = np.array([label_to_int[s] for s in species_list], dtype=int)
    accs, preds = [], {}
    for seed in seeds:
        y_pred = kmedoids_fit(dist, len(label_set), seed)
        accs.append(clustering_accuracy(y_true, y_pred) * 100.0)
        preds[seed] = y_pred
    return dict(n=len(species_list), n_classes=len(label_set), label_set=label_set,
                acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                y_true=y_true, y_pred_seed0=preds[seeds[0]])


def perch_cluster_and_score(X, species_list, seeds=CLUSTERING_SEEDS):
    label_set = sorted(set(species_list))
    label_to_int = {s: i for i, s in enumerate(label_set)}
    y_true = np.array([label_to_int[s] for s in species_list], dtype=int)
    k = len(label_set)

    Xc = l2_normalize(X)
    D = cdist(Xc, Xc, metric="cosine")

    kmedoids_accs, kmedoids_preds = [], {}
    for seed in seeds:
        y_pred = perch_kmedoids(D, k, seed)
        kmedoids_accs.append(perch_clustering_accuracy(y_true, y_pred) * 100.0)
        kmedoids_preds[seed] = y_pred

    kmeans_accs = []
    for seed in seeds:
        y_pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xc)
        kmeans_accs.append(perch_clustering_accuracy(y_true, y_pred) * 100.0)

    return dict(n=len(species_list), n_classes=k, label_set=label_set,
                kmedoids_acc_mean=float(np.mean(kmedoids_accs)), kmedoids_acc_sd=float(np.std(kmedoids_accs)),
                kmeans_acc_mean=float(np.mean(kmeans_accs)), kmeans_acc_sd=float(np.std(kmeans_accs)),
                y_true=y_true, y_pred_seed0=kmedoids_preds[seeds[0]])


def species_breakdown(species_list):
    from collections import Counter
    return dict(Counter(species_list))


def main():
    manifest = list(csv.DictReader(open(EXP_DIR / "manifests" / "full_run_manifest.tsv"), delimiter="\t"))
    conditions = sorted(set(r["condition"] for r in manifest), key=lambda c: (c != "baseline", c))

    all_wav_paths = [r["wav_path"] for r in manifest]
    PERCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[perch] embedding {len(all_wav_paths)} recordings ...")
    perch_cache, perch_stats = embed_recordings(all_wav_paths, "perch_v2_onnx", str(PERCH_CACHE))
    print(f"[perch] done: n_total={perch_stats['n_total']} n_padded={perch_stats['n_padded']}")

    summary_rows = []
    breakdown_rows = []

    for cond in conditions:
        rows = [r for r in manifest if r["condition"] == cond]
        achieved_dbs = [float(r["achieved_db"]) for r in rows if r["achieved_db"]]
        mean_db = float(np.mean(achieved_dbs)) if achieved_dbs else float("nan")

        token_species, token_seqs, seg_rows_all, n_missing_formula = [], [], [], 0
        for r in rows:
            seq, seg_rows = load_formula_data(r["species"], r["recording_name"], cond)
            if seq is None:
                n_missing_formula += 1
                continue
            token_seqs.append(seq)
            token_species.append(r["species"])
            seg_rows_all.extend(seg_rows)

        token_dist = pairwise_edit_distance(token_seqs)
        token_res = cluster_and_score(token_dist, token_species)

        segments_df = pd.DataFrame(seg_rows_all)
        seg_features = compute_segment_features(segments_df)
        rec_features = aggregate_recording_features(seg_features)
        attr_species = rec_features["dataset"].tolist()
        feature_cols = [f"{f}_{stat}" for f in NEW_FEATURES for stat in ("mean", "sd", "min", "max")]
        attr_dist = attribute_distance_matrix(rec_features, feature_cols)
        attr_res = cluster_and_score(attr_dist, attr_species)

        perch_species = [r["species"] for r in rows]
        X = np.stack([perch_cache[r["wav_path"]] for r in rows])
        perch_res = perch_cluster_and_score(X, perch_species)

        shuf_accs = []
        for i in range(N_SHUFFLES):
            rng = np.random.default_rng(SHUFFLE_SEED_BASE + i)
            shuffled = rng.permutation(token_res["y_true"])
            shuf_accs.append(clustering_accuracy(shuffled, token_res["y_pred_seed0"]) * 100.0)
        shuf_mean = float(np.mean(shuf_accs))

        print(f"\n=== {cond} (mean achieved SNR = {mean_db:.2f} dB) ===")
        print(f"  Token:      n={token_res['n']:3d}  classes={token_res['n_classes']}  "
              f"acc={token_res['acc_mean']:5.1f}%  (missing fits: {n_missing_formula})  "
              f"per-species={species_breakdown(token_species)}")
        print(f"  Attributes: n={attr_res['n']:3d}  classes={attr_res['n_classes']}  acc={attr_res['acc_mean']:5.1f}%  "
              f"per-species={species_breakdown(attr_species)}")
        print(f"  Perch:      n={perch_res['n']:3d}  classes={perch_res['n_classes']}  "
              f"k-medoids={perch_res['kmedoids_acc_mean']:5.1f}%  k-means={perch_res['kmeans_acc_mean']:5.1f}%  "
              f"per-species={species_breakdown(perch_species)}")
        print(f"  Shuffled (vs. Token clustering): acc={shuf_mean:5.1f}%")
        print(f"  Majority baseline (fixed, 21/65 of full set): {MAJORITY_BASELINE_PCT:5.1f}%")

        summary_rows.append(dict(
            condition=cond, mean_achieved_db=round(mean_db, 2),
            token_n=token_res["n"], token_n_classes=token_res["n_classes"], token_n_missing=n_missing_formula,
            token_acc_mean=round(token_res["acc_mean"], 1), token_acc_sd=round(token_res["acc_sd"], 1),
            attribute_n=attr_res["n"], attribute_n_classes=attr_res["n_classes"],
            attribute_acc_mean=round(attr_res["acc_mean"], 1), attribute_acc_sd=round(attr_res["acc_sd"], 1),
            perch_n=perch_res["n"], perch_n_classes=perch_res["n_classes"],
            perch_kmedoids_acc_mean=round(perch_res["kmedoids_acc_mean"], 1), perch_kmedoids_acc_sd=round(perch_res["kmedoids_acc_sd"], 1),
            perch_kmeans_acc_mean=round(perch_res["kmeans_acc_mean"], 1), perch_kmeans_acc_sd=round(perch_res["kmeans_acc_sd"], 1),
            shuffled_acc_mean=round(shuf_mean, 1),
            majority_baseline_pct=round(MAJORITY_BASELINE_PCT, 1),
        ))

        for method_name, species_list in [("token", token_species), ("attribute", attr_species), ("perch", perch_species)]:
            for sp, n in sorted(species_breakdown(species_list).items()):
                breakdown_rows.append(dict(condition=cond, method=method_name, species=sp, n=n))

    out_path = EXP_DIR / "results" / "full_run_results.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {out_path}")

    breakdown_path = EXP_DIR / "results" / "full_run_class_breakdown.csv"
    with breakdown_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "method", "species", "n"], lineterminator="\n")
        w.writeheader()
        w.writerows(breakdown_rows)
    print(f"Wrote {breakdown_path}")


if __name__ == "__main__":
    main()

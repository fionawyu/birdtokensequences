"""Inputs for the held-out CV, loaded from the SAME files the published
whole-dataset pipeline reads. Nothing here recomputes features; it only reads:

  token sequences   diagnosis/chapter1_pitch_diagnostic/evaluation_inputs/
                    original_token_sequences.csv           (165 recordings)
  formula segments  diagnosis/chapter1_diagnosis/segments.csv (165/165 coverage,
                    the file contour_dtw_baseline/config.py calls "FIX 1")
  contour DTW       contour_dtw_baseline/distance_matrices/<setting>_dtw_raw.npy
                    + <setting>_keys.csv   (raw, unnormalized -- the variant the
                    project settled on; see contour_dtw_baseline README)
  Perch v2          emb_perch_v2_onnx.npz  (1536-d, one vector per recording)

Why per-recording quantities may be computed once and then split:
edit distance, DTW distance and the formula attributes of a recording are
functions of THAT recording alone (and, for the distances, of one other
recording). They contain no statistic pooled over the dataset and no label.
Restricting such a matrix to the training rows/columns is therefore
mathematically identical to building it from the training recordings only --
a claim `validate_cv.py` checks numerically rather than asserting.
Everything that IS pooled (imputer medians, scaler mean/scale, medoids,
centroids, the cluster->class map) is fitted inside the fold, on training rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from combined_model.common import DATASETS  # noqa: E402

ORIGINAL_SEQUENCES_CSV = (
    REPO_ROOT / "diagnosis" / "chapter1_pitch_diagnostic" / "evaluation_inputs"
    / "original_token_sequences.csv"
)
SEGMENTS_CSV = REPO_ROOT / "diagnosis" / "chapter1_diagnosis" / "segments.csv"
DTW_DIR = REPO_ROOT / "contour_dtw_baseline" / "distance_matrices"
PERCH_NPZ = REPO_ROOT / "emb_perch_v2_onnx.npz"

# NoteMix is excluded here for the same reason contour_dtw_baseline excludes it
# (dropped from the manuscript's main table); the five settings below are the
# five rows of Table tab:quantitative.
SETTINGS = {k: v for k, v in DATASETS.items() if k != "NoteMix"}

SETTING_PAPER_LABEL = {
    "SegMix": "WTSP/fee-bee song seg-mixed",
    "BCCH-var": "BCCH call/song types",
    "Species": "Four species",
    "AllVar": "Species + BCCH variants",
    "YNCCDialect": "yellow cardinal four dialects",
}
SETTING_DIFFICULTY = {
    "SegMix": "Easy", "BCCH-var": "Medium", "Species": "Medium",
    "AllVar": "Hard", "YNCCDialect": "Hard",
}


def recording_id(dataset: str, song_id: str) -> str:
    return f"{dataset}||{song_id}"


# --------------------------------------------------------------------------
def load_token_sequences() -> dict[tuple[str, str], list[str]]:
    df = pd.read_csv(ORIGINAL_SEQUENCES_CSV)
    return {
        (r.dataset, r.song_id): str(r.token_sequence or "").split()
        for r in df.itertuples()
    }


def load_dtw(setting: str) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Raw (unnormalized) contour-DTW distances plus their key order."""
    keys_df = pd.read_csv(DTW_DIR / f"{setting}_keys.csv")
    keys = list(zip(keys_df["dataset"], keys_df["song_id"]))
    matrix = np.load(DTW_DIR / f"{setting}_dtw_raw.npy")
    if matrix.shape != (len(keys), len(keys)):
        raise ValueError(f"{setting}: DTW matrix {matrix.shape} does not match {len(keys)} keys")
    return keys, matrix


def load_perch_by_song_id() -> dict[str, np.ndarray]:
    """emb_perch_v2_onnx.npz is keyed by the absolute audio path of the machine
    it was produced on; resolve by basename, exactly as
    manuscript_rerun_more_seeds/scripts/target1_species65_classification.py does."""
    out: dict[str, np.ndarray] = {}
    with np.load(PERCH_NPZ, allow_pickle=True) as z:
        for full_path in z.files:
            out[Path(full_path).stem] = np.asarray(z[full_path], dtype=float)
    return out


def load_formula_attributes(keys: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    """Per-recording formula attributes (26 columns). Each recording's values
    depend only on its own fitted subformulas, so this is computed once; the
    imputer and scaler that DO pool information are fitted per fold."""
    from combined_model.attribute_model import (
        compute_segment_features, aggregate_recording_features, NEW_FEATURES,
    )
    feature_cols = [f"{f}_{s}" for f in NEW_FEATURES for s in ("mean", "sd", "min", "max")]
    key_set = set(keys)
    seg = pd.read_csv(SEGMENTS_CSV)
    seg = seg[[(d, s) in key_set for d, s in zip(seg["dataset"], seg["song_id"])]]
    rec = aggregate_recording_features(compute_segment_features(seg))
    rec = rec.set_index(["dataset", "song_id"]).loc[keys].reset_index()
    return rec, feature_cols


# --------------------------------------------------------------------------
def build_pool(setting: str) -> dict:
    """The common evaluation pool for one setting.

    The pool is the set of recordings that Token, Shuffled, Trajectory-DTW and
    Formula-attributes can all represent, so those four methods are scored on
    identical splits. For four of the five settings Perch v2 covers exactly the
    same recordings and joins the same pool. For YNCCDialect it does not: the
    cached Perch embeddings for the RN and SL dialects were produced from a
    different curated clip set (Plots_YNCC_2s_curate_1115, "_clip_NNN") than the
    recordings the symbolic pipeline tokenizes ("_split_NNN"), so no recording-
    level correspondence exists. Perch is therefore run on its own pool there
    and flagged -- the same limitation the published table already carries as
    footnote (a), "Perch v2 was evaluated under transfer-domain conditions for
    yellow cardinal".
    """
    datasets = SETTINGS[setting]
    token_seqs = load_token_sequences()
    dtw_keys, dtw_matrix = load_dtw(setting)

    keys = [k for k in dtw_keys if k in token_seqs and token_seqs[k]]
    if len(keys) != len(dtw_keys):
        dropped = [k for k in dtw_keys if k not in keys]
        raise ValueError(f"{setting}: {len(dropped)} DTW keys lack a token sequence: {dropped[:3]}")

    pos = {k: i for i, k in enumerate(dtw_keys)}
    order = [pos[k] for k in keys]
    dtw = dtw_matrix[np.ix_(order, order)]

    label_set = sorted({k[0] for k in keys})
    label_to_int = {d: i for i, d in enumerate(label_set)}
    y = np.array([label_to_int[d] for d, _ in keys], dtype=int)

    rec_features, feature_cols = load_formula_attributes(keys)
    perch = load_perch_by_song_id()
    perch_ok = all(song_id in perch for _, song_id in keys)

    return {
        "setting": setting,
        "keys": keys,
        "recording_ids": [recording_id(d, s) for d, s in keys],
        "y": y,
        "label_set": label_set,
        "class_names": [d.split("_")[2] if len(d.split("_")) > 2 else d for d in label_set],
        "n_classes": len(label_set),
        "token_sequences": [token_seqs[k] for k in keys],
        "dtw": dtw,
        "attributes": rec_features[feature_cols].to_numpy(dtype=float),
        "attribute_cols": feature_cols,
        "perch": np.stack([perch[s] for _, s in keys]) if perch_ok else None,
        "perch_covers_pool": perch_ok,
        "datasets": datasets,
    }


def build_perch_only_pool(setting: str) -> dict | None:
    """Fallback pool for a setting where Perch's cached embeddings cover a
    different recording set (YNCCDialect). Same protocol, its own splits, and
    reported with an explicit n and a footnote rather than silently merged."""
    datasets = SETTINGS[setting]
    prefix_to_dataset = {}
    for d in datasets:
        # 1017_plots47_RNCurate_1017 -> RN
        core = d.split("_")[2]
        prefix_to_dataset[core.replace("Curate", "")] = d

    perch = load_perch_by_song_id()
    report = pd.read_csv(REPO_ROOT / "datasets_link" / "yncc_reconstructed_audio" / "reconstruction_report.csv")
    rows = []
    for r in report.itertuples():
        stem = Path(str(r.wav_path)).stem
        dataset = prefix_to_dataset.get(str(r.label))
        if dataset is None or stem not in perch:
            continue
        rows.append((dataset, stem))
    if not rows:
        return None
    label_set = sorted({d for d, _ in rows})
    label_to_int = {d: i for i, d in enumerate(label_set)}
    return {
        "setting": setting,
        "keys": rows,
        "recording_ids": [recording_id(d, s) for d, s in rows],
        "y": np.array([label_to_int[d] for d, _ in rows], dtype=int),
        "label_set": label_set,
        "class_names": [d.split("_")[2].replace("Curate", "") for d in label_set],
        "n_classes": len(label_set),
        "perch": np.stack([perch[s] for _, s in rows]),
    }

"""Step 3: build DTW distance matrices (contour + segment_means, raw + length-
normalized) per evaluation setting, using dtaidistance's compiled C extension
(dtw.distance_fast / dtw.warping_path_fast). The C extension was not present
in the original pip-installed wheel (a pure-Python fallback was confirmed
working first); it was then built from source in this same venv (gcc module
already loaded) and its output was verified numerically identical to the
pure-Python dtw.distance/dtw.warping_path on test sequences before switching
over -- see README_contour_dtw.md for the note. Using the C path only for
speed (48 cores available; the pure-Python version was projected to take
~8-9 hours across all settings), not changing the DTW algorithm/parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ORIGINAL_SEQUENCES_CSV, CONTOURS_DIR, DIST_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combined_model.common import DATASETS  # noqa: E402

# Scope change (orchestrator, mid-run): drop NoteMix entirely from this baseline.
SETTINGS = {k: v for k, v in DATASETS.items() if k != "NoteMix"}


def sanitize(song_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in song_id)


def load_available_contours() -> dict[tuple[str, str], dict]:
    tok = pd.read_csv(ORIGINAL_SEQUENCES_CSV)
    out = {}
    for _, row in tok.iterrows():
        key = (row["dataset"], row["song_id"])
        path = CONTOURS_DIR / f"{row['dataset']}__{sanitize(row['song_id'])}.npz"
        if path.exists():
            data = np.load(path)
            out[key] = dict(contour=data["contour"], segment_means=data["segment_means"])
    return out


def dtw_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (raw_distance, path_length), via the compiled C extension."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), 0
    raw = dtw.distance_fast(a, b)
    path = dtw.warping_path_fast(a, b)
    return float(raw), len(path)


def build_matrix(keys: list[tuple[str, str]], contours: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(keys)
    raw_mat = np.zeros((n, n), dtype=float)
    norm_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai = np.ascontiguousarray(contours[keys[i]][field], dtype=np.double)
        for j in range(i + 1, n):
            bj = np.ascontiguousarray(contours[keys[j]][field], dtype=np.double)
            raw, path_len = dtw_pair(ai, bj)
            norm = raw / path_len if path_len > 0 else float("nan")
            raw_mat[i, j] = raw_mat[j, i] = raw
            norm_mat[i, j] = norm_mat[j, i] = norm
    return raw_mat, norm_mat


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    contours = load_available_contours()
    print(f"Loaded {len(contours)} recording contours from cache.")

    manifest_rows = []
    for setting, datasets in SETTINGS.items():
        keys = [k for k in contours if k[0] in datasets]
        keys.sort()
        n = len(keys)
        k_classes = len(datasets)
        print(f"\n=== {setting}: n={n} recordings, k={k_classes} classes ===")
        if n == 0:
            print("  SKIPPED: no resolvable recordings for this setting.")
            continue

        raw_c, norm_c = build_matrix(keys, contours, "contour")
        raw_s, norm_s = build_matrix(keys, contours, "segment_means")

        for name, mat in [
            (f"{setting}_dtw_raw", raw_c),
            (f"{setting}_dtw_normalized", norm_c),
            (f"{setting}_segmeans_dtw_raw", raw_s),
            (f"{setting}_segmeans_dtw_normalized", norm_s),
        ]:
            np.save(DIST_DIR / f"{name}.npy", mat)

        # sanity checks
        for label, mat in [("contour", norm_c), ("segment_means", norm_s)]:
            diag_ok = np.allclose(np.diag(mat), 0, atol=1e-8)
            sym_ok = np.allclose(mat, mat.T, equal_nan=True)
            nan_count = np.isnan(mat).sum()
            inf_count = np.isinf(mat).sum()
            print(f"  [{label}] diag~0: {diag_ok}  symmetric: {sym_ok}  nan: {nan_count}  inf: {inf_count}")

        # save keys manifest for this setting (row/col order)
        keys_df = pd.DataFrame(keys, columns=["dataset", "song_id"])
        keys_df.to_csv(DIST_DIR / f"{setting}_keys.csv", index=False)

        manifest_rows.append((setting, n, k_classes))

    manifest_df = pd.DataFrame(manifest_rows, columns=["setting", "n", "k"])
    manifest_df.to_csv(DIST_DIR / "settings_manifest.csv", index=False)
    print("\n=== settings manifest ===")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()

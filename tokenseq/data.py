"""Recording manifests.

A manifest is a CSV file with at least the columns

    recording_id   unique identifier
    class          class label as used in configs/tasks.yaml (e.g. "BCCH/fee-bee")
    path           audio file path, relative to --data-root (or absolute)

`scripts/compute_snr.py` adds a `snr_db` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("recording_id", "class", "path")


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_id": str, "class": str, "path": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["recording_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate recording_id values")
    return df


def audio_path(row: pd.Series, data_root: str | Path | None) -> str:
    p = Path(row["path"])
    return str(p if p.is_absolute() or data_root is None else Path(data_root) / p)


def task_subset(df: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Recordings of a task and their class indices (in the order of `classes`)."""
    sub = df[df["class"].isin(classes)].sort_values("recording_id").reset_index(drop=True)
    index = {c: i for i, c in enumerate(classes)}
    return sub, sub["class"].map(index).to_numpy(dtype=int)

#!/usr/bin/env python3
"""Embed a batch of recordings with Perch v2 (onnx), loading the model once
per batch (not once per file) to amortize the ~33s load cost. Reuses
classiifier.py's embedding convention (5.0s window @ 32kHz, zero-pad short
clips, mean-pool windows to one 1536-d vector per recording) but operates
on our manifest batches instead of the old six-settings CSV format.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

SR = 32000
WIN_S = 5.0
WIN_N = int(round(SR * WIN_S))


def main() -> None:
    manifest_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import librosa
    from perch_hoplite.zoo import model_configs

    t0 = time.time()
    model = model_configs.load_model_by_name("perch_v2_onnx")
    print(f"[model] loaded in {time.time() - t0:.1f}s", flush=True)

    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    embeddings = {}
    for i, row in enumerate(rows, 1):
        key = f"{row['species']}||{row['recording_name']}"
        try:
            audio, _ = librosa.load(row["recording_path"], sr=SR, mono=True)
            if len(audio) < WIN_N:
                audio = np.pad(audio, (0, WIN_N - len(audio)))
            out = model.embed(audio.astype(np.float32))
            emb = np.asarray(out.embeddings).reshape(-1, 1536).mean(axis=0)
            embeddings[key] = emb.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {key}: {exc}", flush=True)
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    np.savez_compressed(out_path, **embeddings)
    print(f"Wrote {out_path} ({len(embeddings)}/{len(rows)} embedded)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Perch v2 recording embeddings (Sec. 3.2.2). Requires requirements-perch.txt.

    python scripts/compute_perch_embeddings.py --manifest data/manifest_snr.csv \
        --data-root data/audio --out outputs/perch_embeddings.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq.data import audio_path, read_manifest  # noqa: E402
from tokenseq.perch import PerchEmbedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="perch_v2", help="perch-hoplite model name")
    args = ap.parse_args()

    df = read_manifest(args.manifest)
    embedder = PerchEmbedder(args.model)
    emb = {}
    for i, (_, r) in enumerate(df.iterrows(), 1):
        emb[r["recording_id"]] = embedder.embed_file(audio_path(r, args.data_root)).astype(np.float32)
        print(f"[{i}/{len(df)}] {r['recording_id']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **emb)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

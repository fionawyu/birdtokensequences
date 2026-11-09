#!/usr/bin/env python3
"""Compute the SNR of every recording (Sec. 3.1.1) and optionally keep only
recordings at or above a threshold (Sec. 3.1.2: SNR >= 10 dB).

    python scripts/compute_snr.py --manifest data/manifest.csv --data-root data/audio \
        --out data/manifest_snr.csv --min-snr 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq.config import load_config  # noqa: E402
from tokenseq.data import audio_path, read_manifest  # noqa: E402
from tokenseq.snr import snr_db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-snr", type=float, default=None,
                    help="drop recordings below this SNR (dB); the paper uses 10")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = read_manifest(args.manifest)
    sr = cfg["audio"]["sample_rate"]
    df["snr_db"] = [snr_db(librosa.load(audio_path(r, args.data_root), sr=sr, mono=True)[0],
                           cfg["snr"]["frame_samples"]) for _, r in df.iterrows()]
    if args.min_snr is not None:
        df = df[df["snr_db"] >= args.min_snr]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} recordings)")


if __name__ == "__main__":
    main()

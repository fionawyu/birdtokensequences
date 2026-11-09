#!/usr/bin/env python3
"""Convert recordings to token sequences (Sec. 2.2-2.6).

For every recording this writes <out>/<recording_id>.json (segments, fitted
formulas, tokens, attributes, trajectories) and <recording_id>_segments.npy
(32 x 32 segment images for the within-token distance).

    python scripts/extract_tokens.py --manifest data/manifest_snr.csv \
        --data-root data/audio --out outputs/tokens --workers 8

    # evaluate formulas at a fixed number of time points instead of one per frame
    python scripts/extract_tokens.py ... --time-points 8
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq import pipeline  # noqa: E402
from tokenseq.config import load_config, parse_time_points, with_overrides  # noqa: E402
from tokenseq.data import audio_path, read_manifest  # noqa: E402


def _work(args):
    path, rid, cfg, out_dir, overwrite = args
    if not overwrite and (out_dir / f"{pipeline.safe_name(rid)}.json").exists():
        return rid, "skipped"
    record, images = pipeline.process_file(path, rid, cfg)
    pipeline.save(record, images, out_dir)
    return rid, " ".join(pipeline.token_sequence(record))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--time-points", default=None,
                    help="'frames' (default: one point per spectrogram frame) or an integer, e.g. 8")
    ap.add_argument("--generations", type=int, default=None, help="override the GA generation count")
    ap.add_argument("--seed", type=int, default=None, help="override the GA seed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tp = parse_time_points(args.time_points) if args.time_points is not None else None
    cfg = with_overrides(cfg, tokens={"time_points": tp},
                         symbolic_regression={"generations": args.generations, "seed": args.seed})
    cfg["tokens"]["time_points"] = parse_time_points(cfg["tokens"]["time_points"])

    out_dir = Path(args.out)
    df = read_manifest(args.manifest)
    jobs = [(audio_path(r, args.data_root), r["recording_id"], cfg, out_dir, args.overwrite)
            for _, r in df.iterrows()]

    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            rid, seq = _work(job)
            print(f"[{i}/{len(jobs)}] {rid}: {seq}", flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futures = [ex.submit(_work, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rid, seq = fut.result()
                print(f"[{i}/{len(jobs)}] {rid}: {seq}", flush=True)


if __name__ == "__main__":
    main()

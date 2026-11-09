#!/usr/bin/env python3
"""Recompute tokens and segment attributes from stored formulas, without
refitting, e.g. to switch between the two time-point modes:

    python scripts/retokenize.py --tokens outputs/tokens --out outputs/tokens_8pts --time-points 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq import pipeline  # noqa: E402
from tokenseq.config import load_config, parse_time_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True, help="directory written by extract_tokens.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--time-points", default=None, help="'frames' or an integer, e.g. 8")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.time_points is not None:
        cfg["tokens"]["time_points"] = args.time_points
    cfg["tokens"]["time_points"] = parse_time_points(cfg["tokens"]["time_points"])

    src, dst = Path(args.tokens), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.json"))
    for f in files:
        record = pipeline.retokenize(json.loads(f.read_text()), cfg)
        (dst / f.name).write_text(json.dumps(record))
        img = f.with_name(f.stem + "_segments.npy")
        if img.exists():
            shutil.copy(img, dst / img.name)
    print(f"retokenized {len(files)} recordings into {dst} (time_points={cfg['tokens']['time_points']})")


if __name__ == "__main__":
    main()

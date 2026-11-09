#!/usr/bin/env python3
"""Within-token distance and entropy-rate gap per class (Sec. 3.3.1, 3.4.5,
Appendix D.2-D.3; Figure 5).

    python scripts/token_metrics.py --manifest data/manifest_snr.csv \
        --tokens outputs/tokens --out outputs/token_metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq import metrics, pipeline  # noqa: E402
from tokenseq.config import load_config  # noqa: E402
from tokenseq.data import read_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed of the permutations")
    args = ap.parse_args()

    ev = load_config(args.config)["evaluation"]
    df = read_manifest(args.manifest)
    rng = np.random.default_rng(args.seed)

    class_rows, token_rows = [], []
    for cls, grp in df.groupby("class", sort=True):
        vecs, toks, gaps = [], [], []
        for rid in grp["recording_id"]:
            rec = pipeline.load(rid, args.tokens)
            imgs = pipeline.load_images(rid, args.tokens)
            vecs.extend(imgs)
            toks.extend(s["token"] for s in pipeline.sound_segments(rec))
            gaps.append(metrics.entropy_rate_gap(pipeline.token_sequence(rec),
                                                 ev["entropy_permutations"], rng))
        wtd = metrics.class_within_token_distance(vecs, toks)
        gaps = np.asarray(gaps, dtype=float)
        class_rows.append({
            "class": cls, "n_recordings": len(grp),
            "within_token_distance": wtd["mean"],
            "entropy_rate_gap": float(np.nanmean(gaps)) if np.isfinite(gaps).any() else np.nan,
        })
        token_rows += [{"class": cls, "token": t, "within_token_distance": d}
                       for t, d in wtd["per_token"].items()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_rows).to_csv(out / "class_metrics.csv", index=False)
    pd.DataFrame(token_rows).to_csv(out / "token_metrics.csv", index=False)
    print(pd.DataFrame(class_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the added-noise conditions of Sec. 3.3.2 (Tables 4 and 11).

Gaussian noise is added to the raw waveform of every recording of a task;
its standard deviation is found by bisection so that the measured SNR falls
within each target range. For every condition this writes the noisy audio
and a manifest pointing to it:

    <out>/<condition>/audio/<recording_id>.wav
    <out>/<condition>/manifest.csv

    python scripts/add_noise.py --manifest data/manifest_snr.csv --data-root data/audio \
        --task four_species --out outputs/noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq.audio import load_audio  # noqa: E402
from tokenseq.config import load_config, load_tasks  # noqa: E402
from tokenseq.data import audio_path, read_manifest, task_subset  # noqa: E402
from tokenseq.pipeline import recording_rng, safe_name  # noqa: E402
from tokenseq.snr import add_noise_to_range  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--task", default="four_species")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks-config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks_config)
    sub, _ = task_subset(read_manifest(args.manifest), tasks[args.task]["classes"])
    sr, ncfg = cfg["audio"]["sample_rate"], cfg["noise"]

    for cond, (low, high) in ncfg["conditions"].items():
        cdir = Path(args.out) / cond
        (cdir / "audio").mkdir(parents=True, exist_ok=True)
        rows = []
        for _, r in sub.iterrows():
            y = load_audio(audio_path(r, args.data_root), sr)
            rng = recording_rng(ncfg["seed"], r["recording_id"])
            noisy, achieved, sigma = add_noise_to_range(y, low, high, rng,
                                                        cfg["snr"]["frame_samples"], ncfg["max_iter"])
            rel = f"audio/{safe_name(r['recording_id'])}.wav"
            sf.write(cdir / rel, noisy.astype(np.float32), sr, subtype="FLOAT")
            rows.append({**r.to_dict(), "path": str((cdir / rel).resolve()),
                         "snr_db": achieved, "noise_sd": sigma})
        pd.DataFrame(rows).to_csv(cdir / "manifest.csv", index=False)
        print(f"{cond}: {len(rows)} recordings -> {cdir}")


if __name__ == "__main__":
    main()

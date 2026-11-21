#!/usr/bin/env python3
"""Build the synthetic reference classes used for the demo's class label.

Each class is one designed song. Its audio is rendered by the demo page's own
synthesizer (node examples/render_sequence.cjs) and sent through the full
pipeline (examples/pipeline_runner.py) with several GA seeds; each run is one
training "recording". As in the paper's held-out evaluation, k-medoids is fitted
on the edit-distance matrix of the training token sequences
(cv_protocol.fit_kmedoids_training), clusters are mapped to classes with the
Hungarian method on training labels (cv_protocol.hungarian_map_from_training),
and the page later assigns new audio to the nearest medoid.

The designed labels are never used to make tokens; they only name the classes.

    python examples/build_references.py            # writes examples/pipeline_references.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pipeline_runner

HERE = Path(__file__).resolve().parent
SEEDS = [47, 11, 23]
# A and B use the same notes in a different order; C has a different inventory.
CLASSES = {
    "A": ["UP", "FLAT", "DOWN", "WIGGLE"],
    "B": ["DOWN", "WIGGLE", "UP", "FLAT"],
    "C": ["UP", "UP", "UP", "FLAT"],
}


def one_run(job):
    name, seed, wav = job
    result = pipeline_runner.run_pipeline(Path(wav), seed, references=None, replay=False)
    return {"class": name, "seed": seed, "tokens": result["token_sequence"],
            "note_tokens": result["note_tokens"], "ga_formulas": result["ga_formulas"],
            "composed_formula": result["composed_formula"],
            "elapsed_seconds": result["elapsed_seconds"]}


def main():
    from cv_protocol import fit_kmedoids_training, hungarian_map_from_training
    from methods import edit_distance_matrix

    started = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        jobs = []
        for name, labels in CLASSES.items():
            wav = Path(scratch) / f"{name}.wav"
            subprocess.run(["node", str(HERE / "render_sequence.cjs"), " ".join(labels), str(wav)],
                           check=True, capture_output=True)
            jobs += [(name, seed, str(wav)) for seed in SEEDS]
        workers = min(len(jobs), os.cpu_count() or 1)
        print(f"Running {len(jobs)} pipeline runs on {workers} processes…", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = []
            for run in pool.map(one_run, jobs):
                print(f"  class {run['class']} seed {run['seed']}: {' '.join(run['note_tokens'])}"
                      f"  ({run['elapsed_seconds']:.0f} s)", flush=True)
                runs.append(run)

    names = list(CLASSES)
    y = np.array([names.index(r["class"]) for r in runs])
    D = edit_distance_matrix([r["tokens"] for r in runs])
    k = len(names)
    clusters, medoids, cost, winning_seed = fit_kmedoids_training(D, k, base_seed=0)
    mapping = hungarian_map_from_training(clusters, y, k, k)
    predicted = np.array([mapping[int(c)] for c in clusters])
    medoid_entries = [{
        "cluster": int(c), "class": names[mapping[c]] if mapping[c] >= 0 else "unassigned",
        "tokens": runs[int(medoids[c])]["tokens"],
        "medoid_run": {"class": runs[int(medoids[c])]["class"], "seed": runs[int(medoids[c])]["seed"]},
    } for c in range(k)]

    payload = {
        "description": "Synthetic song classes for the demo. Tokens come from GA runs on each "
                       "class's reference audio; the designed labels only name the classes.",
        "built_by": "python examples/build_references.py",
        "pipeline_config": pipeline_runner.CONFIG_SCRIPT,
        "classes": {name: {"designed_labels": labels} for name, labels in CLASSES.items()},
        "seeds": SEEDS,
        "training_runs": runs,
        "kmedoids": {"k": k, "cost": float(cost), "winning_seed": int(winning_seed),
                     "training_accuracy": float(np.mean(predicted == y))},
        "medoids": medoid_entries,
    }
    out = HERE / "pipeline_references.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Training accuracy {payload['kmedoids']['training_accuracy']:.2f}; wrote {out} "
          f"in {time.time() - started:.0f} s")
    for m in medoid_entries:
        print(f"  medoid for class {m['class']}: {' '.join(m['tokens'])}")


if __name__ == "__main__":
    main()

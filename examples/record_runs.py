#!/usr/bin/env python3
"""Record real pipeline runs for the hosted demo page.

The hosted page cannot run the genetic program, so it shows these recorded
results instead. Each example song is rendered by the page's own synthesizer
and sent through examples/pipeline_runner.run_pipeline (audio only) with every
seed in CANDIDATE_SEEDS, then classified against the medoids in
pipeline_references.json. None of these seeds was used to build the medoids.

The page shows one run per song: among the runs that reproduce the designed
notes and the song's class, the one with the lowest mean GA loss. The file
keeps which seeds were tried and which were correct, and the page states it.

    python examples/record_runs.py        # writes examples/pipeline_recorded.json
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pipeline_runner
from build_references import CLASSES, SEEDS

HERE = Path(__file__).resolve().parent
CANDIDATE_SEEDS = [37, 5, 61, 71, 83, 97, 101, 113]
assert not set(CANDIDATE_SEEDS) & set(SEEDS), "recorded seeds must differ from the training seeds"
SELECTION_RULE = ("shown run = correct run (designed notes and class reproduced) "
                  "with the lowest mean GA loss")


def one_run(job):
    name, seed, wav = job
    result = pipeline_runner.run_pipeline(Path(wav), seed)
    result["wav"] = f"song {name}"
    return name, seed, result


def mean_loss(result):
    losses = [row["ga_best_loss"] for row in result["segments"] if row["kind"] == "note"]
    return sum(losses) / len(losses)


def main():
    if not pipeline_runner.REFERENCES_JSON.exists():
        raise SystemExit("Run python examples/build_references.py first.")
    started = time.time()
    by_song = {name: [] for name in CLASSES}
    with tempfile.TemporaryDirectory() as scratch:
        jobs = []
        for name, labels in CLASSES.items():
            wav = Path(scratch) / f"{name}.wav"
            subprocess.run(["node", str(HERE / "render_sequence.cjs"), " ".join(labels), str(wav)],
                           check=True, capture_output=True)
            jobs += [(name, seed, str(wav)) for seed in CANDIDATE_SEEDS]
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
            for name, seed, result in pool.map(one_run, jobs):
                c = result["classification"]
                correct = result["note_tokens"] == CLASSES[name] and c["predicted_class"] == name
                print(f"song {name} seed {seed}: {' '.join(result['note_tokens'])} -> class "
                      f"{c['predicted_class']} (distance {c['distance']:g}) {'OK' if correct else 'x'}",
                      flush=True)
                by_song[name].append((seed, correct, result))

    runs, missing = [], []
    for name, tried in by_song.items():
        correct = [(seed, result) for seed, ok, result in tried if ok]
        if not correct:
            missing.append(f"song {name}: " + "; ".join(
                f"seed {seed} -> {' '.join(r['note_tokens'])} (class {r['classification']['predicted_class']})"
                for seed, _, r in tried))
            continue
        seed, result = min(correct, key=lambda pair: mean_loss(pair[1]))
        print(f"song {name}: showing seed {seed} ({len(correct)} of {len(tried)} seeds correct)")
        runs.append({"song": name, "designed_labels": CLASSES[name], "seed": seed,
                     "selection": {"seeds_tried": [s for s, _, _ in tried],
                                   "seeds_correct": [s for s, _ in correct],
                                   "rule": SELECTION_RULE},
                     "result": result})
    if missing:
        raise SystemExit("No correct run for:\n  " + "\n  ".join(missing))

    payload = {
        "description": "Recorded runs of examples/pipeline_runner.py on the demo's example songs.",
        "built_by": "python examples/record_runs.py",
        "runs": runs,
    }
    out = HERE / "pipeline_recorded.json"
    out.write_text(json.dumps(payload) + "\n")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

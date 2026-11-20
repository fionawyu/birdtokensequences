#!/usr/bin/env python3
"""Record real pipeline runs for the hosted demo page.

The hosted page cannot run the genetic program, so it shows these recorded
results instead. Each example song is rendered by the page's own synthesizer
and sent through examples/pipeline_runner.run_pipeline (audio only), then
classified against the medoids in pipeline_references.json. The seed (37) is
not one of the seeds used to build those medoids.

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
RECORD_SEED = 37
assert RECORD_SEED not in SEEDS, "the recorded seed must differ from the training seeds"


def one_run(job):
    name, wav = job
    result = pipeline_runner.run_pipeline(Path(wav), RECORD_SEED)
    result["wav"] = f"song {name}"
    return name, result


def main():
    if not pipeline_runner.REFERENCES_JSON.exists():
        raise SystemExit("Run python examples/build_references.py first.")
    started = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        jobs = []
        for name, labels in CLASSES.items():
            wav = Path(scratch) / f"{name}.wav"
            subprocess.run(["node", str(HERE / "render_sequence.cjs"), " ".join(labels), str(wav)],
                           check=True, capture_output=True)
            jobs.append((name, str(wav)))
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
            runs = []
            for name, result in pool.map(one_run, jobs):
                c = result["classification"]
                print(f"song {name}: {' '.join(result['note_tokens'])} -> class {c['predicted_class']}"
                      f" (distance {c['distance']:g}, {result['elapsed_seconds']:.0f} s)", flush=True)
                runs.append({"song": name, "designed_labels": CLASSES[name], "result": result})

    payload = {
        "description": "Recorded runs of examples/pipeline_runner.py on the demo's example songs.",
        "built_by": "python examples/record_runs.py",
        "seed": RECORD_SEED,
        "runs": runs,
    }
    out = HERE / "pipeline_recorded.json"
    out.write_text(json.dumps(payload) + "\n")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the Chapter 1 pipeline on ONE .wav file and report every step.

Nothing here reimplements the method. Each step calls the function the paper
pipeline uses, in the same order as main_chickeadee_revised.py and mapString.py:

  1. spectrogram + denoising   main_*.process_spectrogram
  2. segmentation              main_*.getSpecSegments_smooth
  3. one GA search per segment main_*.synthesize_frequency -> synthesizers.geneticProgram
  4. composed formula          utils.composeFinalFml
  5. formula -> tokens         mapString.decompose_fmls(mode="silent"), which calls
                               decompose.decomp_fml_string_with_silence and
                               sampleClass.classify_subfmls_by_segments
  6. class label               nearest k-medoid by edit distance, using
                               heldout_cv_evaluation/scripts/{methods,cv_protocol}.py

The input is audio only. The runner never receives note labels, so the tokens
cannot be copied from the designed sequence.

Usage:
    python examples/pipeline_runner.py some.wav [--seed 47] [--json out.json]
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import inspect
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "heldout_cv_evaluation" / "scripts"))

# Parameter set used for the demo: the black-capped chickadee script (2-8 kHz band).
CONFIG_SPECIES = "chickeadee"
CONFIG_SCRIPT = f"main_{CONFIG_SPECIES}_revised.py"
REFERENCES_JSON = Path(__file__).resolve().parent / "pipeline_references.json"

_module = None
_module_lock = threading.Lock()

GA_LINE = re.compile(
    r"^genI: (\d+) \| cur best dataloss: (\S+), cur best fml: (.*) \| (\d+) unique formulae tried$"
)


def pipeline_module():
    """Import main_chickeadee_revised.py without running its batch loop.

    The script creates its output folder at import time, so the import runs in
    a throwaway directory to keep the repository clean."""
    global _module
    with _module_lock:
        if _module is None:
            from run_consistency_recording import load_species_module
            previous = os.getcwd()
            with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
                os.chdir(scratch)
                try:
                    _module = load_species_module(CONFIG_SPECIES)
                finally:
                    os.chdir(previous)
        return _module


def ga_settings():
    """The GA settings actually in force: the defaults of geneticProgram, which
    synthesize_frequency does not override."""
    import synthesizers
    params = inspect.signature(synthesizers.geneticProgram).parameters
    return {name: params[name].default for name in ("popSize", "numGenLim", "fmlSizeLim")}


class GAProgress(io.TextIOBase):
    """Reads the per-generation lines geneticProgram prints (verbose=True is its
    default) and turns them into progress events. Generation 0 starts a new
    segment."""

    def __init__(self, n_segments, n_generations, callback):
        self.n_segments = n_segments
        self.n_generations = n_generations
        self.callback = callback
        self.buffer = ""
        self.segment = 0
        self.generations_seen = []
        self.history = []  # per segment: generation, loss and formula whenever the best loss improves

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = GA_LINE.match(line.strip())
            if not match:
                continue
            generation = int(match.group(1))
            if generation == 0:
                self.segment += 1
                self.generations_seen.append(0)
                self.history.append([])
            self.generations_seen[-1] += 1
            loss, formula = float(match.group(2)), match.group(3)
            steps = self.history[-1]
            if not steps or loss < steps[-1]["loss"]:
                steps.append({"generation": generation, "loss": loss, "formula": formula})
            if self.callback:
                self.callback({
                    "segment": self.segment, "n_segments": self.n_segments,
                    "generation": generation, "n_generations": self.n_generations,
                    "best_loss": float(match.group(2)), "best_formula": match.group(3),
                    "unique_formulas_tried": int(match.group(4)),
                })
        return len(text)


def tokens_via_mapstring(wav_path, fmlfreq):
    """Write the formula file exactly as main_*.py does, then run
    mapString.decompose_fmls on it (the paper's formula-to-token step)."""
    import mapString
    with tempfile.TemporaryDirectory() as run_dir:
        formulae = Path(run_dir) / "west" / "formulae"
        formulae.mkdir(parents=True)
        record = {"file": str(wav_path), "fmlfreq": fmlfreq, "fmlamp": "1"}
        (formulae / (Path(wav_path).stem + "_formulae.yml")).write_text(
            yaml.dump(record, default_flow_style=False, sort_keys=False))
        previous = mapString.BASE_DIR
        mapString.BASE_DIR = Path(run_dir)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, out = mapString.decompose_fmls("west", None, mode="silent", target_fps=None)
        finally:
            mapString.BASE_DIR = previous
    if len(out) != 1:
        raise RuntimeError(f"mapString returned {len(out)} recordings, expected 1")
    return next(iter(out.values()))


def classify_tokens(tokens, references):
    """Nearest training medoid by edit distance, as in the paper's held-out CV."""
    from cv_protocol import predict_nearest_medoid
    from methods import edit_distance_rect
    medoids = references["medoids"]
    D = edit_distance_rect([tokens], [m["tokens"] for m in medoids])
    cluster, dist = predict_nearest_medoid(D)
    return {
        "predicted_class": medoids[int(cluster[0])]["class"],
        "distance": float(dist[0]),
        "medoids": [{"class": m["class"], "tokens": m["tokens"], "distance": float(D[0, i])}
                    for i, m in enumerate(medoids)],
    }


def load_references():
    if REFERENCES_JSON.exists():
        return json.loads(REFERENCES_JSON.read_text())
    return None


def formula_curve(m, fml, seg, times_ext, spec, freqs, duration):
    """Frequency curve of a formula on one segment, using the mapping of
    synthesizers.fmlDrawSpec: evaluate at the segment's times, map each value to
    the nearest row of linspace(0, uyFreqlim/12000, height), skip row 0."""
    from funcWrappers import wrapExtEvalRPN
    from utils import lispToRPN
    a, b = int(seg[0]), int(seg[1])
    width, height = b - a, spec.shape[0]
    time_seg = times_ext[0:width]
    normed_times = np.linspace(time_seg[0], time_seg[-1], width)
    normed_freqs = np.linspace(0, m.uyFreqlim / 12000, height)
    rpn = lispToRPN(fml)
    if rpn is None:
        return [], []
    pred, _ = wrapExtEvalRPN(rpn, normed_times)
    with np.errstate(invalid="ignore"):
        rows = np.abs(normed_freqs[None, :] - np.asarray(pred, dtype=float)[:, None]).argmin(axis=1)
    frame_s = duration / spec.shape[1]
    step = max(1, int(np.ceil(width / 160)))
    columns = [k for k in range(0, width, step) if rows[k] != 0]
    curve = [[round((a + k) * frame_s, 4), round(float(freqs[rows[k]]), 1)] for k in columns]
    return curve, [(k, int(rows[k])) for k in columns]


def replay_data(m, record, segments, subfmls, times_ext, history, check):
    """Everything the page needs to replay the run: the pipeline's denoised
    spectrogram as an image, and for each segment the GA search history with
    the curve of each improving formula."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    import matplotlib.image as mpimg
    from synthesizers import fmlDrawSpec
    spec, freqs, duration = record["denoised_spec"], record["freqs"], record["duration"]
    image = cm.viridis(np.clip(spec[::2, ::2], 0, 1))[::-1]  # top row = highest frequency
    buffer = io.BytesIO()
    mpimg.imsave(buffer, image, format="png")
    out = []
    for i, seg in enumerate(segments):
        steps = list(history[i])
        if not steps or steps[-1]["formula"] != subfmls[i]:
            steps.append({"generation": steps[-1]["generation"] if steps else 0,
                          "loss": None, "formula": subfmls[i]})
        for step in steps:
            step["curve"], drawn = formula_curve(m, step["formula"], seg, times_ext, spec, freqs, duration)
        # the curve shown for the final formula must be what the paper's fmlDrawSpec draws
        a, b = int(seg[0]), int(seg[1])
        image_seg = fmlDrawSpec(subfmls[i], times_ext[0:b - a], spec[:, a:b], m.lyFreqlim, m.uyFreqlim, m.thickness)
        check(f"replay curve of segment {i + 1} matches fmlDrawSpec",
              all(image_seg[row, k] == 1 for k, row in drawn))
        out.append({"start_s": a * duration / spec.shape[1], "end_s": b * duration / spec.shape[1],
                    "history": steps})
    return {"spectrogram_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "duration_s": duration, "freq_lo_hz": float(freqs[0]), "freq_hi_hz": float(freqs[-1]),
            "segments": out}


def run_pipeline(wav_path, seed=47, on_progress=None, references="default", replay=True):
    import synthesizers
    from sampleClass import classify_subfml_with_rule, seg_W_fixed
    m = pipeline_module()
    settings = ga_settings()
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    m.set_seed(seed)
    synthesizers.resetGuardStats()
    started = time.time()

    # 1-2. spectrogram, denoising, segmentation (same calls as main_*.py)
    with contextlib.redirect_stdout(io.StringIO()):
        record = m.process_spectrogram(wav_path, m.noise_threshold, m.amp_threshold,
                                       m.lyFreqlim, m.uyFreqlim)
        spec, freqs, duration = record["denoised_spec"], record["freqs"], record["duration"]
        frame_ms = (duration / spec.shape[1]) * 1000.0
        segments, _mask, _score = m.getSpecSegments_smooth(
            spec, ampTol_hi=m.ampTol_hi, ampTol_lo=m.ampTol_lo,
            min_gap_frames=int(round(m.min_gap_ms / frame_ms)),
            min_dur_frames=int(round(m.min_dur_ms / frame_ms)),
            quantile=m.quantile, use_quantile=m.use_quantile)
    if not segments:
        raise ValueError("The pipeline found no sound segments in this audio.")

    # 3. one GA search per segment; times are stretched by times_extended as in main_*.py
    times_ext = np.linspace(0, duration * m.times_extended, spec.shape[1], endpoint=False)
    progress = GAProgress(len(segments), settings["numGenLim"], on_progress)
    with contextlib.redirect_stdout(progress):
        subfmls = m.synthesize_frequency(spec, times_ext, segments, m.time_per_seg, freqs,
                                         m.lyFreqlim, m.uyFreqlim, m.thickness)
    check("one GA search ran per segment", progress.segment == len(segments) == len(subfmls))
    check(f"each GA search ran {settings['numGenLim']} generations",
          progress.generations_seen == [settings["numGenLim"]] * len(segments))
    ga_losses = [s["best_loss"] for s in synthesizers.GUARD_STATS["segments"]]
    check("a GA loss was recorded for each segment", len(ga_losses) == len(segments))

    # 4. composed formula
    fmlfreq = m.composeFinalFml(segments, subfmls, times_ext)

    # 5. formula -> tokens through mapString
    decomp = tokens_via_mapstring(wav_path, fmlfreq)
    seg_ext, sub_list, labels = decomp["segments_freq"], decomp["freq_subfmls"], decomp["freq_labels"]

    rows, note_index = [], 0
    for (t1, t2), sub, label in zip(seg_ext, sub_list, labels):
        W, _ = seg_W_fixed(t1, t2)
        explained, rule = classify_subfml_with_rule(sub, W, start_sec=t1, end_sec=t2)
        check(f"rule explanation agrees with the label for {t1:.3f}-{t2:.3f}", explained == label)
        is_note = sub != "0" and note_index < len(subfmls) and sub == subfmls[note_index]
        rows.append({
            "kind": "note" if is_note else "gap",
            "start_s": t1 / m.times_extended, "end_s": t2 / m.times_extended,
            "start_t": t1, "end_t": t2,
            "formula": sub, "token": label, "rule": rule,
            "ga_best_loss": ga_losses[note_index] if is_note and note_index < len(ga_losses) else None,
        })
        if is_note:
            note_index += 1
    check("every GA formula appears, in order, in the decomposed formula", note_index == len(subfmls))
    check("token count equals decomposed segment count", len(labels) == len(seg_ext))

    tokens = [str(t) for t in labels]
    note_tokens = [r["token"] for r in rows if r["kind"] == "note"]

    # 6. class label
    if references == "default":
        references = load_references()
    classification = classify_tokens(tokens, references) if references else None

    result = {
        "wav": str(wav_path),
        "parameters": {
            "config": CONFIG_SCRIPT, "seed": seed,
            "lyFreqlim": m.lyFreqlim, "uyFreqlim": m.uyFreqlim, "thickness": m.thickness,
            "noise_threshold": m.noise_threshold, "ampTol_hi": m.ampTol_hi, "ampTol_lo": m.ampTol_lo,
            "min_dur_ms": m.min_dur_ms, "min_gap_ms": m.min_gap_ms,
            "times_extended": m.times_extended, **settings,
            "candidates_evaluated": int(synthesizers.GUARD_STATS["evaluated"]),
            "candidates_rejected_by_derivative_guard": int(synthesizers.GUARD_STATS["rejected"]),
        },
        "audio": {"duration_s": duration, "sample_rate_used": record["sample_rate"],
                  "spectrogram_shape": list(spec.shape), "frame_ms": frame_ms},
        "ga_segments_frames": [[int(a), int(b)] for a, b in segments],
        "ga_formulas": list(subfmls),
        "composed_formula": fmlfreq,
        "segments": rows,
        "token_sequence": tokens,
        "note_tokens": note_tokens,
        "classification": classification,
        "checks": checks,
        "elapsed_seconds": time.time() - started,
    }
    if replay:
        result["replay"] = replay_data(m, record, segments, subfmls, times_ext, progress.history, check)
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise AssertionError("Pipeline consistency checks failed: " + "; ".join(failed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--json", help="write the full result to this file")
    args = parser.parse_args()

    def show(p):
        if p["generation"] in (0, p["n_generations"] - 1):
            print(f"  segment {p['segment']}/{p['n_segments']} generation {p['generation'] + 1}: "
                  f"loss {p['best_loss']:.4f}  {p['best_formula']}", flush=True)

    result = run_pipeline(Path(args.wav).resolve(), args.seed, on_progress=show, replay=False)
    print(f"\nDuration: {result['audio']['duration_s']:.3f} s, {len(result['ga_formulas'])} segments")
    for row in result["segments"]:
        print(f"  {row['start_s']:.3f}-{row['end_s']:.3f} s  {row['token']:<6}  {row['formula']}")
        print(f"      rule: {row['rule']}")
    print("Token sequence:", " ".join(result["token_sequence"]))
    if result["classification"]:
        c = result["classification"]
        print(f"Predicted class: {c['predicted_class']} (edit distance {c['distance']:g})")
    print(f"Checks passed: {len(result['checks'])}/{len(result['checks'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the existing per-species formula-fitting pipeline on ONE recording,
repeated across several seeds, to measure cross-seed consistency.

This reuses each species' own main_{SPECIES}_revised.py functions and
constants unchanged (process_spectrogram, getSpecSegments_smooth,
synthesize_frequency, composeFinalFml, set_seed, and all species-specific
thresholds such as lyFreqlim/noise_threshold/min_dur_ms). The genetic
programming core (synthesizers.geneticProgram) is not touched here.

Only the plotting / .mat bookkeeping from the original scripts' `__main__`
batch loop is skipped -- it is irrelevant to consistency and would multiply
storage and runtime across thousands of repeated runs.

Output per recording:
    {output_dir}/{species}/{recording_name}/seed_{seed}/formula.yml
    {output_dir}/{species}/{recording_name}/seed_{seed}/DONE

A seed is only ever (re)run if its DONE marker is missing, so a task that
dies partway through its 4 seeds can be resubmitted and will continue from
the first incomplete seed instead of repeating finished ones.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [47, 11, 23, 37]


def default_output_dir() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "consistency_results"
    return REPO_ROOT / "results_consistency"


def default_runtime_log_dir() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "consistency_runtime_logs"
    return REPO_ROOT / "runtime_logs"


def load_species_module(species: str):
    """Import main_{species}_revised.py as a module without running its
    __main__ batch loop (that block is guarded by `if __name__ == "__main__"`,
    so importing only defines functions/constants and calls set_seed(47) once
    at import time -- harmless, since we call set_seed() again per seed below).
    """
    script = REPO_ROOT / f"main_{species}_revised.py"
    if not script.exists():
        raise FileNotFoundError(f"No pipeline script for species {species}: {script}")

    # The script does `rngseed = int(sys.argv[1]) if len(sys.argv) >= 2 else 47`
    # at module scope. Hide our own argv (--audio, --species, ...) from it.
    saved_argv = sys.argv
    try:
        sys.argv = [str(script)]
        spec = importlib.util.spec_from_file_location(f"_pipeline_{species}", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    return module


def atomic_write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, path)  # atomic rename on the same filesystem
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def run_one_seed(module, species: str, audio_path: Path, seed: int, seed_dir: Path) -> dict:
    import synthesizers  # the instance the species module already imported
    module.set_seed(seed)
    synthesizers.resetGuardStats()

    t0 = time.time()
    record = module.process_spectrogram(
        audio_path, module.noise_threshold, module.amp_threshold,
        module.lyFreqlim, module.uyFreqlim,
    )
    denoised_spec = record["denoised_spec"]
    freqs = record["freqs"]
    duration = record["duration"]

    frame_ms = (duration / denoised_spec.shape[1]) * 1000.0
    min_dur_frames = int(round(module.min_dur_ms / frame_ms))
    min_gap_frames = int(round(module.min_gap_ms / frame_ms))

    segments, _mask, _a = module.getSpecSegments_smooth(
        denoised_spec,
        ampTol_hi=module.ampTol_hi,
        ampTol_lo=module.ampTol_lo,
        min_gap_frames=min_gap_frames,
        min_dur_frames=min_dur_frames,
        quantile=module.quantile,
        use_quantile=module.use_quantile,
    )

    duration_ext = duration * module.times_extended
    times_ext = np.linspace(0, duration_ext, denoised_spec.shape[1], endpoint=False)

    subfreqfmls = module.synthesize_frequency(
        denoised_spec, times_ext, segments, module.time_per_seg, freqs,
        module.lyFreqlim, module.uyFreqlim, module.thickness,
    )
    fmlfreq = module.composeFinalFml(segments, subfreqfmls, times_ext)
    elapsed = time.time() - t0

    result = {
        "audio_path": str(audio_path),
        "species": species,
        "seed": seed,
        "n_segments": len(segments),
        "subfreqfmls": subfreqfmls,
        "fmlfreq": fmlfreq,
        "elapsed_seconds": elapsed,
        # derivative guard (synthesizers.DERIV_GUARD): candidates whose
        # formula, f' or f'' is not finite and real at the sampled time points
        # score negative infinity; per-segment entries follow segment order
        "deriv_guard": {
            "enabled": bool(synthesizers.DERIV_GUARD),
            "candidates_evaluated": int(synthesizers.GUARD_STATS["evaluated"]),
            "candidates_rejected": int(synthesizers.GUARD_STATS["rejected"]),
            "init_redrawn": int(synthesizers.GUARD_STATS["init_redrawn"]),
            "segments": list(synthesizers.GUARD_STATS["segments"]),
        },
    }
    atomic_write_yaml(seed_dir / "formula.yml", result)
    return {"n_segments": len(segments), "elapsed_seconds": elapsed}


def process_recording(
    species: str, audio_path: Path, output_dir: Path, seeds: list[int],
    runtime_log_dir: Path | None,
) -> bool:
    recording_name = audio_path.stem
    rec_dir = output_dir / species / recording_name
    module = None  # lazily imported only if at least one seed needs running

    task_start = time.time()
    seed_results = []
    any_failed = False

    for seed in seeds:
        seed_dir = rec_dir / f"seed_{seed}"
        done_marker = seed_dir / "DONE"

        if done_marker.exists():
            print(f"[skip] {species}/{recording_name} seed={seed} already DONE")
            seed_results.append({"seed": seed, "status": "skipped_done"})
            continue

        if module is None:
            module = load_species_module(species)

        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            stats = run_one_seed(module, species, audio_path, seed, seed_dir)
            done_marker.write_text(f"completed_at={time.time()}\n")
            seed_results.append({"seed": seed, "status": "ok", **stats})
            print(
                f"[ok] {species}/{recording_name} seed={seed} "
                f"segments={stats['n_segments']} elapsed={stats['elapsed_seconds']:.1f}s"
            )
        except Exception as e:  # noqa: BLE001 - deliberately broad: one bad
            # recording/seed must not take down the whole array task, and we
            # want it recorded as failed (no DONE marker) so a resubmission
            # retries exactly this seed.
            any_failed = True
            seed_results.append({"seed": seed, "status": "failed", "error": str(e)})
            print(f"[FAIL] {species}/{recording_name} seed={seed}: {e}", file=sys.stderr)
            traceback.print_exc()

    task_elapsed = time.time() - task_start
    log_entry = {
        "species": species,
        "recording": recording_name,
        "audio_path": str(audio_path),
        "start_time": task_start,
        "end_time": time.time(),
        "total_elapsed_seconds": task_elapsed,
        "seeds": seed_results,
        "status": "failed" if any_failed else "ok",
    }
    if runtime_log_dir is not None:
        runtime_log_dir_species = runtime_log_dir / species
        runtime_log_dir_species.mkdir(parents=True, exist_ok=True)
        log_path = runtime_log_dir_species / f"{recording_name}.json"
        log_path.write_text(json.dumps(log_entry, indent=2))

    print(
        f"[{'FAIL' if any_failed else 'DONE'}] {species}/{recording_name} "
        f"total={task_elapsed:.1f}s"
    )
    return not any_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="path to the recording (.wav)")
    parser.add_argument("--species", required=True, help="species code, e.g. TEWA")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--runtime-log-dir", default=str(default_runtime_log_dir()))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"seeds to run (default: {DEFAULT_SEEDS}, matching config.py's "
             "CONSISTENCY_FORMULA_SEEDS)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        sys.exit(f"[ERROR] audio file not found: {audio_path}")

    output_dir = Path(args.output_dir)
    runtime_log_dir = Path(args.runtime_log_dir) if args.runtime_log_dir else None

    ok = process_recording(args.species, audio_path, output_dir, args.seeds, runtime_log_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

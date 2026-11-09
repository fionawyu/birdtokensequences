"""Step 1-2: extract per-frame dominant-frequency contours for the Contour+DTW
baseline, from real audio via the project's own utils.py preprocessing
(load_and_normalize_audio -> create_audio_spectrogram -> advanced_denoise_spectrogram),
restricted to non-silent segments from segments.csv (FIX 1 path), with segment
start/end times corrected from the stretched formula-fitting timebase back to
real seconds (FIX 2: divide by times_extended).

No formulas, no tokens (except to identify SILENCE_TOKENS segments to
exclude), no fitted-ridge values anywhere in this script -- frequency values
come only from argmax over the real, re-computed denoised spectrogram.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_and_normalize_audio, create_audio_spectrogram, advanced_denoise_spectrogram  # noqa: E402
from config import (  # noqa: E402
    ORIGINAL_SEQUENCES_CSV, SEGMENTS_CSV, CONTOURS_DIR, SILENCE_TOKENS,
    SPECIES_PARAMS, TIMES_EXTENDED, resolve_audio_path,
)


def sanitize(song_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in song_id)


def extract_one(dataset: str, song_id: str, wav_path: Path, seg_rows: pd.DataFrame) -> dict | None:
    params = SPECIES_PARAMS[dataset]
    times_extended = TIMES_EXTENDED[dataset]

    sample_rate, audio_data = load_and_normalize_audio(str(wav_path))
    spectrogram, freqs_khz = create_audio_spectrogram(audio_data, sample_rate)
    denoised = advanced_denoise_spectrogram(spectrogram, params["noise"], params["amp"])

    window_size = int(sample_rate * 5 / 1000)
    hop_size = window_size // 4
    n_frames = spectrogram.shape[1]
    times = np.arange(n_frames) * hop_size / sample_rate

    ly_khz = params["ly"] / 1000.0
    uy_khz = params["uy"] / 1000.0
    band_mask = (freqs_khz >= ly_khz) & (freqs_khz <= uy_khz)
    band_freqs = freqs_khz[band_mask]

    seg_rows = seg_rows.sort_values("start")

    contour_vals = []
    segment_means = []
    n_segments_used = 0
    n_segments_dropped_silent = 0
    n_segments_dropped_out_of_range = 0
    n_frames_dropped_zero = 0

    for _, srow in seg_rows.iterrows():
        if srow["token"] in SILENCE_TOKENS:
            n_segments_dropped_silent += 1
            continue

        real_start = srow["start"] / times_extended
        real_end = srow["end"] / times_extended

        mask_t = (times >= real_start) & (times < real_end)
        idxs = np.flatnonzero(mask_t)
        if len(idxs) == 0:
            n_segments_dropped_out_of_range += 1
            continue

        seg_vals = []
        for i in idxs:
            col = denoised[band_mask, i]
            if np.all(col == 0):
                n_frames_dropped_zero += 1
                continue
            freq = band_freqs[np.argmax(col)]
            seg_vals.append(freq)
            contour_vals.append(freq)

        if len(seg_vals) == 0:
            n_segments_dropped_out_of_range += 1  # all frames zero-thresholded away
            continue

        segment_means.append(float(np.mean(seg_vals)))
        n_segments_used += 1

    if n_segments_used == 0:
        return None

    return dict(
        contour=np.array(contour_vals, dtype=float),
        segment_means=np.array(segment_means, dtype=float),
        n_segments_used=n_segments_used,
        n_segments_dropped_silent=n_segments_dropped_silent,
        n_segments_dropped_out_of_range=n_segments_dropped_out_of_range,
        n_frames_dropped_zero=n_frames_dropped_zero,
    )


def main() -> None:
    CONTOURS_DIR.mkdir(parents=True, exist_ok=True)
    tok = pd.read_csv(ORIGINAL_SEQUENCES_CSV)
    seg = pd.read_csv(SEGMENTS_CSV)

    resolved = 0
    unresolved_rows = []
    empty_rows = []
    t0 = time.time()

    for i, row in tok.iterrows():
        dataset, song_id = row["dataset"], row["song_id"]
        wav_path, note = resolve_audio_path(dataset, song_id, row["file_path"])
        if wav_path is None:
            unresolved_rows.append((dataset, song_id, note))
            print(f"[{i+1}/{len(tok)}] UNRESOLVED {dataset} / {song_id}: {note}")
            continue

        seg_rows = seg[(seg["dataset"] == dataset) & (seg["song_id"] == song_id)]
        try:
            result = extract_one(dataset, song_id, wav_path, seg_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i+1}/{len(tok)}] ERROR {dataset} / {song_id}: {exc}")
            unresolved_rows.append((dataset, song_id, f"exception: {exc}"))
            continue

        if result is None:
            print(f"[{i+1}/{len(tok)}] EMPTY (no usable non-silent segments) {dataset} / {song_id}")
            empty_rows.append((dataset, song_id))
            continue

        out_path = CONTOURS_DIR / f"{dataset}__{sanitize(song_id)}.npz"
        np.savez(
            out_path,
            contour=result["contour"],
            segment_means=result["segment_means"],
            n_segments_used=result["n_segments_used"],
            n_segments_dropped_silent=result["n_segments_dropped_silent"],
            n_segments_dropped_out_of_range=result["n_segments_dropped_out_of_range"],
            n_frames_dropped_zero=result["n_frames_dropped_zero"],
        )
        resolved += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(tok):
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(tok)}] progress: resolved={resolved} elapsed={elapsed:.1f}s")

    print("\n=== extract_contours summary ===")
    print(f"total canonical recordings: {len(tok)}")
    print(f"resolved+extracted: {resolved}")
    print(f"unresolved (no audio): {len(unresolved_rows)}")
    for u in unresolved_rows:
        print("  ", u)
    print(f"resolved but empty (no usable non-silent frames): {len(empty_rows)}")
    for e in empty_rows:
        print("  ", e)

    summary_path = CONTOURS_DIR.parent / "extract_contours_summary.csv"
    pd.DataFrame(
        unresolved_rows, columns=["dataset", "song_id", "reason"]
    ).to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

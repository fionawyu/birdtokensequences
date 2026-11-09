"""Build combined per-song audio for the YNCC RN/SL dialect datasets.

Root cause (confirmed by the orchestrating session): no combined per-song
audio exists anywhere in the repo for RN/SL. The only source is
`yncc/yncc_curated_data_process/Plots_YNCC_2s_curate_1115/{RN,SL}/*.mat`,
which holds fixed ~2-second spectrogram-magnitude clips named
`<song_prefix>_clip_<NNN>.mat`, not one-file-per-song like Co/U's
`<song>_combined.mat`.

This script groups those clips by `<song_prefix>` (the part before
`_clip_NNN`), reconstructs each clip's audio via the existing
`mat_to_audio.reconstruct_one()` (same Griffin-Lim procedure, N_FFT=512,
SAMPLE_RATE=36000 -- reused unmodified, not reimplemented), sorts by clip
index, and concatenates the clip waveforms in order into one continuous
per-song-prefix .wav file under
`contour_dtw_baseline/yncc_stitched_audio/{RN,SL}/<song_prefix>.wav`.

This is a NAIVE concatenation (no crossfade/overlap correction) of
independently Griffin-Lim-reconstructed clips -- an extra layer of
lossiness beyond Co/U's single-file reconstruction. Documented as a caveat
in README_contour_dtw.md.

segments.csv/original_token_sequences.csv song_ids for RN/SL are
`<song_prefix>_split_<NNN>` (each split is its own canonical "recording"
with its own segment/token labels). Multiple splits of the same prefix
share the same stitched audio file; resolution strips the `_split_NNN`
suffix to find it (see resolve_audio.py).

Only clip prefixes that actually have >=1 source .mat file are stitched;
song_ids whose prefix has NO clips anywhere in the source directory cannot
be resolved at all and are reported (not silently dropped) by the caller.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mat_to_audio import reconstruct_one, SAMPLE_RATE  # noqa: E402

SRC_ROOT = REPO_ROOT / "yncc" / "yncc_curated_data_process" / "Plots_YNCC_2s_curate_1115"
OUT_ROOT = Path(__file__).resolve().parent / "yncc_stitched_audio"

CLIP_RE = re.compile(r"^(?P<prefix>.+)_clip_(?P<idx>\d+)$")


def group_clips(label: str) -> dict[str, list[tuple[int, Path]]]:
    in_dir = SRC_ROOT / label
    groups: dict[str, list[tuple[int, Path]]] = {}
    for mat_path in sorted(in_dir.glob("*.mat")):
        m = CLIP_RE.match(mat_path.stem)
        if not m:
            print(f"WARNING: unexpected filename pattern, skipping: {mat_path}")
            continue
        prefix = m.group("prefix")
        idx = int(m.group("idx"))
        groups.setdefault(prefix, []).append((idx, mat_path))
    for prefix in groups:
        groups[prefix].sort(key=lambda t: t[0])
    return groups


def main() -> None:
    report_rows = []
    for label in ("RN", "SL"):
        groups = group_clips(label)
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {label}: {len(groups)} song prefixes from {sum(len(v) for v in groups.values())} clips ===")
        for prefix, clips in sorted(groups.items()):
            audio_parts = []
            clip_indices = []
            for idx, mat_path in clips:
                audio, info = reconstruct_one(mat_path)
                audio_parts.append(audio)
                clip_indices.append(idx)
            stitched = np.concatenate(audio_parts)
            out_path = out_dir / f"{prefix}.wav"
            sf.write(str(out_path), stitched, SAMPLE_RATE)
            dur = len(stitched) / SAMPLE_RATE
            print(f"  {prefix}: clips {clip_indices} -> {out_path.name} dur={dur:.3f}s")
            report_rows.append((label, prefix, clip_indices, dur))

    print("\nDone. Stitched song prefixes:")
    for label, prefix, idxs, dur in report_rows:
        print(f"  [{label}] {prefix} clips={idxs} duration={dur:.3f}s")
    print(f"\nTotal stitched files: {len(report_rows)}")


if __name__ == "__main__":
    main()

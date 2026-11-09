"""Segmentation into sound segments and silence intervals (Sec. 2.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: int          # first frame (inclusive)
    end: int            # last frame (exclusive)
    kind: str           # "sound" or "silence"

    @property
    def n_frames(self) -> int:
        return self.end - self.start


def frame_energy(spec: np.ndarray, percentile: float) -> np.ndarray:
    """Per-frame percentile of amplitude across frequency bins, scaled to [0, 1]."""
    e = np.percentile(spec, percentile, axis=0)
    lo, hi = float(e.min()), float(e.max())
    return (e - lo) / (hi - lo) if hi > lo else np.zeros_like(e)


def hysteresis(energy: np.ndarray, onset: float, offset: float) -> np.ndarray:
    """A frame becomes active when energy rises above `onset` and stays active
    until energy falls below `offset`."""
    active = np.zeros(energy.shape, dtype=bool)
    on = False
    for i, v in enumerate(energy):
        if not on and v > onset:
            on = True
        elif on and v < offset:
            on = False
        active[i] = on
    return active


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment(spec: np.ndarray, frame_s: float, cfg: dict) -> list[Interval]:
    """Return the ordered sound segments and the silence intervals between them.

    Segments shorter than `min_segment_ms` are discarded; neighbouring
    segments separated by a gap shorter than `min_gap_ms` are merged.
    Leading and trailing silence is not part of the sequence.
    """
    energy = frame_energy(spec, cfg["frame_percentile"])
    active = hysteresis(energy, cfg["onset"], cfg["offset"])

    min_len = cfg["min_segment_ms"] / 1000.0 / frame_s
    runs = [(s, e) for s, e in _runs(active) if (e - s) >= min_len]

    min_gap = cfg["min_gap_ms"] / 1000.0 / frame_s
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[Interval] = []
    for i, (s, e) in enumerate(merged):
        if i > 0:
            out.append(Interval(int(merged[i - 1][1]), int(s), "silence"))
        out.append(Interval(int(s), int(e), "sound"))
    return out

"""Segment attributes (Sec. 2.6) and recording-level attribute vectors (Sec. 3.2.1).

Formula values v are positions in the band-limited spectrogram scaled to
[0, 1]; they convert to frequency as  Hz = f_lo + v * (f_hi - f_lo).
Fitting time is real time multiplied by `time_scale`, so derivatives with
respect to real time pick up factors of `time_scale` and `time_scale ** 2`.
"""

from __future__ import annotations

import numpy as np

from . import formula as F
from .tokens import time_points

ATTRIBUTES = (
    "duration_s",
    "start_khz", "end_khz", "min_khz", "max_khz", "mean_khz", "range_khz",
    "slope_khz_per_s", "curvature_khz_per_s2", "linear_dev_khz",
)
STATS = ("mean", "std", "min", "max")


def segment_attributes(formula: str, n_frames: int, frame_s: float, time_scale: float,
                       band_hz: tuple[float, float], mode: str | int = "frames") -> dict:
    node = F.parse(formula)
    t = time_points(n_frames, frame_s, time_scale, mode)
    f0, d1, d2 = F.sympy_derivatives(node, t)

    khz_per_unit = (band_hz[1] - band_hz[0]) / 1000.0
    khz = band_hz[0] / 1000.0 + f0 * khz_per_unit
    t_real = t / time_scale

    if t_real.size >= 2 and np.all(np.isfinite(khz)):
        line = np.polyval(np.polyfit(t_real, khz, 1), t_real)
        lin_dev = float(np.sqrt(np.mean((khz - line) ** 2)))
    else:
        lin_dev = 0.0

    return {
        "duration_s": n_frames * frame_s,
        "start_khz": float(khz[0]),
        "end_khz": float(khz[-1]),
        "min_khz": float(np.min(khz)),
        "max_khz": float(np.max(khz)),
        "mean_khz": float(np.mean(khz)),
        "range_khz": float(np.max(khz) - np.min(khz)),
        "slope_khz_per_s": float(np.mean(d1) * khz_per_unit * time_scale),
        "curvature_khz_per_s2": float(np.mean(np.abs(d2)) * khz_per_unit * time_scale ** 2),
        "linear_dev_khz": lin_dev,
    }


def recording_vector(segments: list[dict]) -> np.ndarray:
    """mean, SD, min and max of each attribute over sound segments, plus the
    segment count: 10 * 4 + 1 = 41 dimensions."""
    feats = []
    for name in ATTRIBUTES:
        v = np.array([s[name] for s in segments], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            feats.extend([np.nan] * len(STATS))
        else:
            feats.extend([v.mean(), v.std(), v.min(), v.max()])
    feats.append(float(len(segments)))
    return np.asarray(feats, dtype=float)


def recording_vector_names() -> list[str]:
    return [f"{a}_{s}" for a in ATTRIBUTES for s in STATS] + ["n_segments"]

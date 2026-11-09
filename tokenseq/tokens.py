"""Mapping formulas to trajectory-shape tokens (Sec. 2.5, Table 8).

Time points. A formula is evaluated at uniformly spaced time points spanning
the segment duration. Two modes are supported:

* ``"frames"`` (default, as in the manuscript): one point per spectrogram
  frame of the segment, i.e. the same points the formula was fitted on;
* an integer ``n`` (e.g. ``8``): ``n`` uniformly spaced points, independent
  of segment length.
"""

from __future__ import annotations

import numpy as np

from . import formula as F

UP, DOWN, FLAT, WIGGLE, SILENCE = "UP", "DOWN", "FLAT", "WIGGLE", "SILENCE"
SHAPE_TOKENS = (UP, DOWN, FLAT, WIGGLE)
ALL_TOKENS = SHAPE_TOKENS + (SILENCE,)


def time_points(n_frames: int, frame_s: float, time_scale: float,
                mode: str | int = "frames") -> np.ndarray:
    """Evaluation times (on the scaled time axis used during fitting)."""
    duration = n_frames * frame_s * time_scale
    n = n_frames if mode == "frames" else int(mode)
    return np.linspace(0.0, duration, max(n, 1), endpoint=False)


def _is_zero(node: F.Node, y: np.ndarray, cfg: dict) -> bool:
    if isinstance(node, str) and node in cfg["zero_constants"]:
        return True
    finite = y[np.isfinite(y)]
    return finite.size > 0 and bool(np.all(np.abs(finite) < cfg["near_zero"]))


def shape_token(y: np.ndarray, t: np.ndarray, cfg: dict) -> str:
    """Apply the UP / DOWN / FLAT / WIGGLE rules to sampled formula values."""
    ok = np.isfinite(y)
    y, t = y[ok], t[ok]
    if y.size < 2:
        return FLAT
    rng = float(y.max() - y.min())
    if rng < cfg["flat_range"]:
        return FLAT

    slope = np.gradient(y, t)
    signs = np.sign(slope)
    signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] * signs[:-1] < 0))
    if reversals >= cfg["min_sign_reversals"] and rng >= cfg["range_cutoff"]:
        return WIGGLE

    change = float(y[-1] - y[0])
    directional = abs(change) >= cfg["change_fraction"] * rng
    if change > 0 and np.mean(slope > 0) >= cfg["slope_sign_fraction"] and directional:
        return UP
    if change < 0 and np.mean(slope < 0) >= cfg["slope_sign_fraction"] and directional:
        return DOWN
    return FLAT if rng < cfg["range_cutoff"] else WIGGLE


def formula_token(formula: str, n_frames: int, frame_s: float, time_scale: float,
                  cfg: dict) -> str:
    """Token of the formula fitted to a sound segment of `n_frames` frames."""
    node = F.parse(formula)
    t = time_points(n_frames, frame_s, time_scale, cfg["time_points"])
    y = F.evaluate(node, t)
    if _is_zero(node, y, cfg):
        return SILENCE
    return shape_token(y, t, cfg)

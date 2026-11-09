"""Spectrogram preprocessing (Sec. 2.2)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class Spectrogram:
    """Band-limited, thresholded magnitude spectrogram.

    values: (n_freq, n_frames) amplitudes in [0, 1]
    freqs_hz: centre frequency of each row
    frame_s: duration of one frame (hop length) in seconds
    """

    values: np.ndarray
    freqs_hz: np.ndarray
    frame_s: float

    @property
    def n_frames(self) -> int:
        return self.values.shape[1]

    def frame_times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.frame_s


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Load a recording as mono, resampled to `sample_rate`, peak-normalised."""
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / peak if peak > 0 else y).astype(np.float64)


def compute_spectrogram(y: np.ndarray, cfg: dict) -> Spectrogram:
    """STFT -> fixed bandpass (row selection) -> percentile noise threshold.

    The noise threshold sets every amplitude below the `noise_percentile`-th
    percentile of the non-zero amplitudes of its frame to zero.
    """
    sr = cfg["sample_rate"]
    win = int(round(sr * cfg["window_ms"] / 1000.0))
    hop = max(1, int(round(win * cfg["hop_fraction"])))
    n_fft = max(int(cfg["n_fft"]), win)
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                              window="hann", center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    lo, hi = cfg["band_hz"]
    band = (freqs >= lo) & (freqs <= hi)
    mag, freqs = mag[band], freqs[band]

    out = np.zeros_like(mag)
    for j in range(mag.shape[1]):
        frame = mag[:, j]
        nz = frame[frame > 0]
        if nz.size:
            thr = np.percentile(nz, cfg["noise_percentile"])
            out[:, j] = np.where(frame >= thr, frame, 0.0)
    peak = out.max()
    if peak > 0:
        out /= peak
    return Spectrogram(values=out, freqs_hz=freqs, frame_s=hop / sr)


def dominant_frequency(spec: np.ndarray) -> np.ndarray:
    """Row index of the highest-amplitude bin in every frame of `spec`."""
    return np.argmax(spec, axis=0)

"""Perch v2 recording embeddings (Sec. 3.2.2).

Recordings are converted to mono, resampled to 32 kHz and processed in
non-overlapping 5-s windows (shorter recordings are zero-padded). The
1,536-dimensional window embeddings are averaged into one vector per
recording. Requires `perch-hoplite` (see requirements-perch.txt).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 32000
WINDOW_S = 5.0
EMBED_DIM = 1536


class PerchEmbedder:
    def __init__(self, model_name: str = "perch_v2"):
        from perch_hoplite.zoo import model_configs

        self.model = model_configs.load_model_by_name(model_name)

    def embed_waveform(self, y: np.ndarray) -> np.ndarray:
        win = int(round(SAMPLE_RATE * WINDOW_S))
        n_win = max(1, int(np.ceil(len(y) / win)))
        y = np.pad(np.asarray(y, dtype=np.float32), (0, n_win * win - len(y)))
        windows = [self.model.embed(y[i * win:(i + 1) * win]).embeddings for i in range(n_win)]
        return np.concatenate([np.asarray(e).reshape(-1, EMBED_DIM) for e in windows]).mean(axis=0)

    def embed_file(self, path: str) -> np.ndarray:
        import librosa

        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return self.embed_waveform(y)

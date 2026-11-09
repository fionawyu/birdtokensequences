"""End-to-end smoke test on a synthetic two-note recording."""

import copy

import numpy as np

from tokenseq import pipeline
from tokenseq.config import load_config, parse_time_points


def synthetic_song(sr=44100):
    rng = np.random.default_rng(0)
    t = np.arange(int(0.8 * sr)) / sr
    y = 1e-3 * rng.standard_normal(len(t))
    a = (t > 0.10) & (t < 0.25)                       # rising note
    tt = t[a] - 0.10
    y[a] += np.sin(2 * np.pi * (3000 * tt + 0.5 * 20000 * tt ** 2))
    b = (t > 0.45) & (t < 0.60)                       # flat note
    y[b] += np.sin(2 * np.pi * 5000 * t[b])
    return y


def test_end_to_end(tmp_path):
    cfg = load_config()
    cfg["symbolic_regression"]["generations"] = 30
    cfg["tokens"]["time_points"] = parse_time_points(cfg["tokens"]["time_points"])
    rec, imgs = pipeline.process_waveform(synthetic_song(), "synthetic", cfg)

    kinds = [iv["kind"] for iv in rec["intervals"]]
    assert kinds == ["sound", "silence", "sound"]
    assert imgs.shape == (2, 32 * 32)
    seq = pipeline.token_sequence(rec)
    assert seq[1] == "SILENCE"
    for s in pipeline.sound_segments(rec):
        assert len(s["attributes"]) == 10
        assert cfg["audio"]["band_hz"][0] / 1000 - 0.2 <= s["attributes"]["mean_khz"] <= 11.2

    pipeline.save(rec, imgs, tmp_path)
    assert pipeline.load("synthetic", tmp_path) == rec

    cfg8 = copy.deepcopy(cfg)
    cfg8["tokens"]["time_points"] = 8
    rec8 = pipeline.retokenize(rec, cfg8)
    assert rec8["time_points"] == 8
    assert len(pipeline.token_sequence(rec8)) == len(seq)

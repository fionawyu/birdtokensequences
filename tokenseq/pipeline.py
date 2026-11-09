"""Recording -> segments -> formulas -> tokens and attributes (Sec. 2.2-2.6)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from . import audio, segmentation, symbolic_regression, tokens
from .attributes import segment_attributes


def recording_rng(seed: int, recording_id: str) -> np.random.Generator:
    """Per-recording generator, independent of processing order."""
    return np.random.default_rng([int(seed), zlib.crc32(recording_id.encode())])


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of a 2-D array to size x size."""
    h, w = img.shape
    rr, cc = np.meshgrid(np.linspace(0, h - 1, size), np.linspace(0, w - 1, size), indexing="ij")
    return map_coordinates(img, [rr, cc], order=1, mode="nearest")


def process_waveform(y: np.ndarray, recording_id: str, cfg: dict) -> tuple[dict, np.ndarray]:
    """Run the full pipeline on a waveform (already at cfg['audio']['sample_rate']).

    Returns (record, segment_images). `record` is JSON-serialisable;
    `segment_images` holds one flattened size x size spectrogram slice per
    sound segment (Sec. 3.3.1), in the order of the sound segments.
    """
    sr_cfg, tok_cfg = cfg["symbolic_regression"], cfg["tokens"]
    spec = audio.compute_spectrogram(y, cfg["audio"])
    band = (float(spec.freqs_hz[0]), float(spec.freqs_hz[-1]))
    n_rows = spec.values.shape[0]
    rng = recording_rng(sr_cfg["seed"], recording_id)
    img_size = cfg["evaluation"]["segment_image_size"]

    intervals, images = [], []
    for k, iv in enumerate(segmentation.segment(spec.values, spec.frame_s, cfg["segmentation"])):
        item = {
            "index": k, "kind": iv.kind,
            "start_frame": int(iv.start), "end_frame": int(iv.end), "n_frames": iv.n_frames,
            "start_s": iv.start * spec.frame_s, "end_s": iv.end * spec.frame_s,
        }
        if iv.kind == "silence":
            item["token"] = tokens.SILENCE
            intervals.append(item)
            continue

        sl = spec.values[:, iv.start:iv.end]
        rows = audio.dominant_frequency(sl)
        traj = rows / (n_rows - 1)                                   # position in [0, 1]
        t = np.arange(iv.n_frames) * spec.frame_s * sr_cfg["time_scale"]
        fit = symbolic_regression.fit(t, traj, sr_cfg, rng)

        item.update(
            formula=fit.formula, loss=fit.loss, formula_size=fit.size,
            trajectory=traj.tolist(),
            trajectory_khz=(spec.freqs_hz[rows] / 1000.0).tolist(),
            token=tokens.formula_token(fit.formula, iv.n_frames, spec.frame_s,
                                       sr_cfg["time_scale"], tok_cfg),
            attributes=segment_attributes(fit.formula, iv.n_frames, spec.frame_s,
                                          sr_cfg["time_scale"], band),
        )
        intervals.append(item)
        images.append(resize(sl, img_size).ravel())

    record = {
        "recording_id": recording_id,
        "frame_s": spec.frame_s,
        "band_hz": list(band),
        "n_frames": spec.n_frames,
        "intervals": intervals,
    }
    imgs = np.stack(images) if images else np.zeros((0, img_size * img_size))
    return record, imgs


def process_file(path: str, recording_id: str, cfg: dict) -> tuple[dict, np.ndarray]:
    y = audio.load_audio(path, cfg["audio"]["sample_rate"])
    return process_waveform(y, recording_id, cfg)


def retokenize(record: dict, cfg: dict) -> dict:
    """Recompute tokens and attributes of a stored record from its formulas
    (no refitting), e.g. with different mapping thresholds."""
    tok_cfg, scale = cfg["tokens"], cfg["symbolic_regression"]["time_scale"]
    out = json.loads(json.dumps(record))
    for iv in out["intervals"]:
        if iv["kind"] != "sound":
            continue
        iv["token"] = tokens.formula_token(iv["formula"], iv["n_frames"], record["frame_s"],
                                           scale, tok_cfg)
        iv["attributes"] = segment_attributes(iv["formula"], iv["n_frames"], record["frame_s"],
                                              scale, tuple(record["band_hz"]))
    return out


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def safe_name(recording_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in recording_id)


def save(record: dict, images: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(record["recording_id"])
    (out_dir / f"{name}.json").write_text(json.dumps(record))
    np.save(out_dir / f"{name}_segments.npy", images.astype(np.float32))


def load(recording_id: str, out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / f"{safe_name(recording_id)}.json").read_text())


def load_images(recording_id: str, out_dir: Path) -> np.ndarray:
    return np.load(Path(out_dir) / f"{safe_name(recording_id)}_segments.npy")


def token_sequence(record: dict) -> list[str]:
    return [iv["token"] for iv in record["intervals"]]


def sound_segments(record: dict) -> list[dict]:
    return [iv for iv in record["intervals"] if iv["kind"] == "sound"]

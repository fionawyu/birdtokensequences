"""Reconstruct audio from the YNCC normalized-magnitude spectrogram .mat files
via Griffin-Lim phase estimation, for feeding into a raw-audio model (Perch 2).

SOURCE DATA
    yncc/yncc_curated_data_process/Plots_YNCC_2s_curate_1115/{SL,Co,RN,U}/*.mat
Each .mat holds:
    X          (129, T) float64, magnitude normalized to [0, 1] per-clip (linear, not dB)
    freqs_khz  (1, 129) float64, 1.0-10.0 kHz
    times_sec  (1, T)   float64, 0.0-2.0 s

CAVEAT (confirmed with the user before writing this): these are the actual
working unit for this pipeline (fixed ~2s clips), not full songs. No raw
audio exists anywhere in the repo for YNCC, and phase is not recoverable —
Griffin-Lim reconstruction is a lossy approximation with known artifacts.
Treat downstream Perch 2 results on this dataset as exploratory, not
comparable 1:1 to the other (real-audio) settings in results_perch_v2.md.

SYNTHESIS PARAMETERS
    freqs_khz step = 70.3125 Hz. Solving step = sample_rate / n_fft for a
    standard power-of-two n_fft gives an EXACT match at n_fft=512,
    sample_rate=36000 (36000/512 = 70.3125 exactly), and the resulting bin
    range [14:143) is exactly 129 bins wide -- matching X's row count
    exactly. This is treated as recovered ground truth for the STFT grid,
    not a free choice. hop_length is fit from the times_sec spacing.
    Bins outside the 1-10 kHz band (which the source pipeline evidently
    discarded/filtered) are zero-filled before inversion.
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.io as sio
import soundfile as sf

SRC = Path("yncc/yncc_curated_data_process/Plots_YNCC_2s_curate_1115")
OUT = Path("datasets_link/yncc_reconstructed_audio")

N_FFT = 512
SAMPLE_RATE = 36000
FREQ_BIN_HZ = SAMPLE_RATE / N_FFT  # 70.3125, matches measured freqs_khz step exactly
N_FREQ_BINS = N_FFT // 2 + 1       # 257

LABELS = ["SL", "Co", "RN", "U"]


def reconstruct_one(mat_path: Path):
    m = sio.loadmat(mat_path)
    X = m["X"].astype(np.float64)                      # (129, T)
    freqs_khz = m["freqs_khz"].flatten()
    times_sec = m["times_sec"].flatten()

    n_frames = X.shape[1]
    clip_duration_s = float(times_sec[-1] - times_sec[0])
    hop_length = max(1, int(round(clip_duration_s / (n_frames - 1) * SAMPLE_RATE))) if n_frames > 1 else N_FFT // 4

    low_bin = int(round(freqs_khz[0] * 1000 / FREQ_BIN_HZ))
    high_bin = low_bin + X.shape[0]  # exclusive

    full_mag = np.zeros((N_FREQ_BINS, n_frames), dtype=np.float64)
    full_mag[low_bin:high_bin, :] = X

    audio = librosa.griffinlim(
        full_mag,
        n_fft=N_FFT,
        hop_length=hop_length,
        n_iter=100,
        momentum=0.8,
        random_state=42,
    )
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    title = str(m.get("title", [mat_path.stem])[0]) if "title" in m else mat_path.stem

    return audio, {
        "mat_path": str(mat_path),
        "title": title,
        "n_freq_bins_source": int(X.shape[0]),
        "low_bin": low_bin,
        "high_bin": high_bin,
        "n_frames": n_frames,
        "hop_length": hop_length,
        "clip_duration_s_source": clip_duration_s,
        "clip_duration_s_reconstructed": len(audio) / SAMPLE_RATE,
        "peak_before_norm": float(peak),
        "rms": float(np.sqrt(np.mean(audio**2))),
    }


def main():
    rows = []
    for label in LABELS:
        in_dir = SRC / label
        out_dir = OUT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        mat_files = sorted(in_dir.glob("*.mat"))
        for mat_path in mat_files:
            audio, info = reconstruct_one(mat_path)
            wav_path = out_dir / (mat_path.stem + ".wav")
            sf.write(str(wav_path), audio, SAMPLE_RATE)
            info.update({"label": label, "wav_path": str(wav_path)})
            rows.append(info)
            print(f"[{label}] {mat_path.name} -> {wav_path.name}  "
                  f"dur={info['clip_duration_s_reconstructed']:.3f}s rms={info['rms']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reconstruction_report.csv", index=False)
    print(f"\nReconstructed {len(df)} clips -> {OUT}")
    print(df.groupby("label").size())
    print("\nDuration source vs reconstructed (should match closely):")
    print(df[["clip_duration_s_source", "clip_duration_s_reconstructed"]].describe())
    print("\nRMS distribution (flag near-zero = likely silent/failed reconstruction):")
    print(df["rms"].describe())
    n_silent = (df["rms"] < 1e-4).sum()
    print(f"\nNear-silent reconstructions (rms<1e-4): {n_silent}/{len(df)}")


if __name__ == "__main__":
    main()

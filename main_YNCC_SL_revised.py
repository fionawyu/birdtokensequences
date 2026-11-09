# -*- coding: utf-8 -*-
# Compatible with .wav and .mat inputs for your spectrogram/FML pipeline

from pathlib import Path
import os, sys, json
import numpy as np
import yaml
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import librosa
import librosa.display
from time import time

# Your custom libs (assumed available in PYTHONPATH)
from synthesizers import geneticProgram, geneticProgramAmp, fmlDrawSpec, ampflmDraw, set_seed
from utils import composeFinalFml, load_and_normalize_audio, create_audio_spectrogram, \
                  advanced_denoise_spectrogram
from fml2latex import fml2latex

from scipy.ndimage import gaussian_filter

sys.setrecursionlimit(20000)

# ----------------------------
# Config (edit these as needed)
# ----------------------------
data_path = "./Plots_YNCC_2s_curate_1115/SL"
output_path = "1017_plots{seed}_YNCC_SL_1017_minDur{min_dur}_minGap{min_gap}_hi{hi}_lo{lo}_ext{ext}x_10_modified_grammar"

# Constant parameters
lyFreqlim = 1000         # Hz
uyFreqlim = 10000        # Hz
thickness = 2            # drawing thickness (pixels)
ampTol = 0.25            # legacy
amp_threshold = 0.25
noise_threshold = 0.1
time_per_seg = 2.5       # seconds "budget" per segment for GP
front_row = 1            # visualize detected segments on top rows
ampTol_hi = 0.15
ampTol_lo = 0.05
min_dur_ms = 3
min_gap_ms = 0
quantile = 0.95
use_quantile = False
times_extended = 10
freq_extended = 5

# debug
debug_size = 0

def plotSaveResults(file_path, audio_data, duration, denoised_spec, predicted_spec,
                    longest_duration, output_file, output_folder, lyFreqlim, uyFreqlim,
                    freqs, verbose=True, smooth_pred=False, smooth_sigma=0.5):
    """
    Plot and save Denoised + Predicted spectrograms with correct orientation.
    """
    freq_times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)
    freqs_khz = freqs / 1000.0

    pred_to_plot = predicted_spec
    if smooth_pred:
        pred_to_plot = gaussian_filter(predicted_spec, sigma=float(smooth_sigma))

    # ---- 2-panel figure: denoised + predicted ----
    fig = plt.figure(figsize=(15, 8))

    # Create extent for proper axis scaling
    extent = [freq_times[0], freq_times[-1], freqs_khz[0], freqs_khz[-1]]

    # Denoised
    ax1 = plt.subplot(211)
    img1 = ax1.imshow(
        denoised_spec,
        aspect='auto',
        origin='upper',
        extent=extent,
        cmap='viridis',
        interpolation='nearest'
    )
    ax1.set_title("Denoised Spectrogram")
    ax1.set_ylabel("Frequency (kHz)")
    ax1.set_ylim(lyFreqlim/1000.0, uyFreqlim/1000.0)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    plt.colorbar(img1, ax=ax1, label="Magnitude")

    # Predicted
    ax2 = plt.subplot(212)
    img2 = ax2.imshow(
        pred_to_plot,
        aspect='auto',
        origin='upper',
        extent=extent,
        cmap='viridis',
        interpolation='nearest'
    )
    ax2.set_title("Predicted Spectrogram")
    ax2.set_ylabel("Frequency (kHz)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(lyFreqlim/1000.0, uyFreqlim/1000.0)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    plt.colorbar(img2, ax=ax2, label="Magnitude")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        os.makedirs(output_folder, exist_ok=True)
        base_name = Path(file_path).stem

        # Denoised-only
        fig_denoised = plt.figure(figsize=(15, 4))
        ax = plt.axes()
        img = ax.imshow(
            denoised_spec,
            aspect='auto',
            origin='upper',
            extent=extent,
            cmap='viridis',
            interpolation='nearest'
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_title("Denoised Spectrogram")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_ylim(lyFreqlim/1000.0, uyFreqlim/1000.0)
        plt.colorbar(img, label="Magnitude")
        denoised_folder = os.path.join(output_folder, "denoised")
        os.makedirs(denoised_folder, exist_ok=True)
        denoised_path = os.path.join(denoised_folder, base_name + "_denoised.png")
        plt.savefig(denoised_path, dpi=300, bbox_inches="tight")
        plt.close(fig_denoised)

        # Predicted-only
        fig_predicted = plt.figure(figsize=(15, 4))
        ax = plt.axes()
        img = ax.imshow(
            pred_to_plot,
            aspect='auto',
            origin='upper',
            extent=extent,
            cmap='viridis',
            interpolation='nearest'
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_title("Predicted Spectrogram")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_ylim(lyFreqlim/1000.0, uyFreqlim/1000.0)
        plt.colorbar(img, label="Magnitude")
        predicted_folder = os.path.join(output_folder, "predicted")
        os.makedirs(predicted_folder, exist_ok=True)
        predicted_path = os.path.join(predicted_folder, base_name + "_predicted.png")
        plt.savefig(predicted_path, dpi=300, bbox_inches="tight")
        plt.close(fig_predicted)


# ----------------------------
# Utility: segments (smoothed)
# ----------------------------
def _mask_to_runs(mask):
    x = mask.astype(int)
    diff1 = np.r_[x[0], np.diff(x)]
    starts = np.where(diff1 == 1)[0]
    diff2 = np.r_[np.diff(x), -x[-1]]
    ends   = np.where(diff2 == -1)[0] + 1
    return starts, ends

def getSpecSegments_smooth(
    spec,
    ampTol_hi,
    ampTol_lo=None,
    min_gap_frames=40,
    min_dur_frames=72,
    quantile=0.95,
    use_quantile=False
):
    # per-column score
    if use_quantile:
        col_score = np.quantile(spec, quantile, axis=0)
    else:
        col_score = spec.max(axis=0)

    # robust normalize 0..1
    lo, hi = np.percentile(col_score, [10, 99])
    a = (col_score - lo) / (hi - lo + 1e-8)
    a = np.clip(a, 0.0, 1.0)

    # hysteresis
    if ampTol_lo is None:
        ampTol_lo = 0.6 * ampTol_hi
    on = False
    mask = np.zeros_like(a, dtype=bool)
    for i in range(len(a)):
        v = a[i]
        if (not on) and (v >= ampTol_hi): on = True
        elif on and (v < ampTol_lo):      on = False
        mask[i] = on

    # drop short segments
    s, e = _mask_to_runs(mask)
    for i in range(len(s)):
        if (e[i] - s[i]) < int(min_dur_frames):
            mask[s[i]:e[i]] = False

    # merge small gaps
    s, e = _mask_to_runs(mask)
    for i in range(len(e) - 1):
        gap = s[i+1] - e[i]
        if gap <= int(min_gap_frames):
            mask[e[i]:s[i+1]] = True

    starts, ends = _mask_to_runs(mask)
    segPairs = list(zip(starts, ends))
    print("Number of Segment Pairs:", len(segPairs))
    return segPairs, mask, a

# ----------------------------
# Cut spectrogram by freq band
# ----------------------------
def cut_spectrogram(denoised_spec, freqs, lyFreqlim, uyFreqlim):
    # 找到频段索引
    lyFreqind = 0
    for i in range(len(freqs)):
        if (freqs[i] >= lyFreqlim):
            lyFreqind = i
            break

    uyFreqind = len(freqs) - 1
    for i in range(len(freqs) - 1, -1, -1):
        if (freqs[i] <= uyFreqlim):
            uyFreqind = i
            break

    denoised_spec = denoised_spec[lyFreqind:uyFreqind + 1, :]
    freqs = freqs[lyFreqind:uyFreqind + 1]

    return denoised_spec, freqs

# ------------------------------------
# Loaders: .wav and .mat (unified API)
# ------------------------------------
def _load_mat_as_record(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim):
    """
    Read a .mat file that contains a spectrogram and (optionally) freqs/times/audio/fs.
    Returns a dict with: denoised_spec, freqs, times, duration, sample_rate, audio_data, average_freq.
    """
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # spectrogram keys to try
    spec_keys = ["denoised_spec", "spec", "S", "spectrogram", "X", "Mag", "power_spectrogram"]
    spec = None
    for k in spec_keys:
        if k in mat:
            spec = np.array(mat[k])
            break
    if spec is None:
        raise ValueError(f"No spectrogram-like key found in {file_path}. Tried: {spec_keys}")

    # ensure [freq, time]
    if spec.ndim != 2:
        raise ValueError("Spectrogram must be 2D [freq, time].")
    # heuristic transpose if suspicious
    if spec.shape[0] < spec.shape[1] and spec.shape[0] < 16:
        spec = spec.T

    # audio (optional)
    audio = None
    for k in ["audio", "y", "waveform", "x"]:
        if k in mat:
            audio = np.array(mat[k]).astype(float).ravel()
            break

    # sample rate (optional)
    fs = None
    for k in ["fs", "Fs", "sr", "sample_rate"]:
        if k in mat:
            fs = float(mat[k])
            break

    # frequency axis (optional)
    freqs = None
    for k in ["freqs", "f", "F"]:
        if k in mat:
            freqs = np.array(mat[k]).astype(float).ravel()
            break
    if freqs is None and fs is not None:
        freqs = np.linspace(0, fs/2, spec.shape[0])

    # time axis (optional)
    times = None
    for k in ["times", "t", "time", "T"]:
        if k in mat:
            times = np.array(mat[k]).astype(float).ravel()
            if times.size == spec.shape[1] and times.max() > 1000 and fs is not None:
                times = times / fs
            break

    # duration
    if audio is not None and fs is not None and audio.size > 0:
        duration = audio.size / fs
    else:
        if times is not None and times.size > 0:
            duration = float(times[-1]) if times[-1] > 0 else 1.0
        else:
            duration = 1.0

    # compute denoised (unless already denoised_spec by name)
    if "denoised_spec" in mat:
        denoised_spec = spec
    else:
        denoised_spec = advanced_denoise_spectrogram(
            spec, noise_threshold, amp_threshold,
            normalize_time=False, binarize=False
        )

    # axes best-effort
    if freqs is None:
        freqs = np.linspace(0, 10000, denoised_spec.shape[0])  # 0..10k Hz fallback
    if times is None:
        times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)

    # trim by band + ensure ascending
    denoised_spec, freqs = cut_spectrogram(denoised_spec, freqs, lyFreqlim, uyFreqlim)

    sample_rate = fs if fs is not None else 2 * freqs[-1] if freqs is not None else 22050.0
    audio_data = audio if audio is not None else np.zeros(int(duration * sample_rate), dtype=float)
    average_freq = float(np.mean(freqs)) if freqs is not None else 0.0

    record = {
        "file": str(file_path),
        "denoised_spec": denoised_spec,
        "freqs": freqs,
        "times": times,
        "duration": duration,
        "sample_rate": sample_rate,
        "audio_data": audio_data,
        "average_freq": average_freq,
    }
    return record

def load_prepare_spec(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim):
    """
    Unified loader:
      - .wav → normalize + spectrogram + denoise
      - .mat → parse arrays (plus optional audio/freq/time)
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".mat":
        return _load_mat_as_record(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim)

    # WAV path
    sample_rate, audio_data = load_and_normalize_audio(file_path)
    original_spec, _ = create_audio_spectrogram(audio_data, sample_rate)
    denoised_spec = advanced_denoise_spectrogram(
        original_spec, noise_threshold, amp_threshold,
        normalize_time=False, binarize=False
    )
    duration = len(audio_data) / sample_rate
    freqs = np.linspace(0, sample_rate / 2, denoised_spec.shape[0])

    # trim by band + ensure ascending
    denoised_spec, freqs = cut_spectrogram(denoised_spec, freqs, lyFreqlim, uyFreqlim)
    times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)
    average_freq = np.mean(freqs)

    record = {
        "file": str(file_path),
        "denoised_spec": denoised_spec,
        "freqs": freqs,
        "times": times,
        "duration": duration,
        "sample_rate": sample_rate,
        "audio_data": audio_data,
        "average_freq": average_freq,
    }
    return record

# --------------------------------------
# Synthesis helpers (freq & amplitude)
# --------------------------------------
def synthesize_frequency(denoised_spec, times, segments, time_per_seg, freqs, lyFreqlim, uyFreqlim, thickness=5):
    subfreqfmls = []
    for seg in segments:
        spec_seg = denoised_spec[:, seg[0]:seg[1]]
        time_seg = times[0:seg[1] - seg[0]]
        _ = np.mean(freqs)
        fmlfreq, _score = geneticProgram(time_seg, spec_seg, time_per_seg, thickness, lyFreqlim, uyFreqlim, fmlSeed = "")
        subfreqfmls.append(fmlfreq)
    assert len(subfreqfmls) == len(segments)
    return subfreqfmls

def drawRelativeSpec(segments, subfmls, times, denoised_spec, lyFreqlim, uyFreqlim, thickness):
    assert len(subfmls) == len(segments), "number of subfmls does not match number of segments"
    pred_spec = np.zeros(denoised_spec.shape)
    for segI, seg in enumerate(segments):
        time_seg = times[0:seg[1] - seg[0]]
        freq_spec = fmlDrawSpec(subfmls[segI], time_seg, denoised_spec[:,seg[0]:seg[1]], lyFreqlim, uyFreqlim, thickness)
        pred_spec[:, seg[0]:seg[1]] = freq_spec
    return pred_spec

def synthesize_amplitude(denoised_spec, times, freq_spec, segments, time_per_seg, freqs, lyFreqlim, uyFreqlim):
    # not used in current run (fmlamp="1"), but fixed to accept `segments`
    subampfmls = []
    for seg in segments:
        spec_seg = denoised_spec[:, seg[0]:seg[1]]
        time_seg = times[seg[0]:seg[1]]
        freq_spec_seg = freq_spec[:, seg[0]:seg[1]]
        _ = np.mean(freqs)
        fmlamp, _score = geneticProgramAmp(time_seg, spec_seg, freq_spec_seg, time_per_seg, lyFreqlim, uyFreqlim, fmlSeed = "")
        subampfmls.append(fmlamp)
    assert len(subampfmls) == len(segments)
    return subampfmls

# --------------------------------------
# Main
# --------------------------------------
def main():
    rngseed = int(sys.argv[1]) if len(sys.argv) >= 2 else 47
    set_seed(rngseed)

    out_dir = output_path.format(seed=rngseed,
                                 min_dur=min_dur_ms, min_gap=min_gap_ms,
                                 hi=ampTol_hi, lo=ampTol_lo, ext=times_extended)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    root = Path(data_path)

    # gather all unique directories containing .wav or .mat
    all_subdirs = set()
    for file_path in root.glob("**/*"):
        if file_path.suffix.lower() in [".wav", ".mat"]:
            all_subdirs.add(file_path.parent)

    # write parameters JSON
    parameters = {
        'lyFreqlim': lyFreqlim,
        'uyFreqlim': uyFreqlim,
        'thickness': thickness,
        'ampTol': ampTol,
        'amp_threshold': amp_threshold,
        'noise_threshold': noise_threshold,
        'time_per_seg': time_per_seg,
        'front_row': front_row,
        'ampTol_hi': ampTol_hi,
        'ampTol_lo': ampTol_lo,
        'min_gap_ms': min_gap_ms,
        'min_dur_ms': min_dur_ms,
        'quantile': quantile,
        'use_quantile': use_quantile,
        'times_extended': times_extended,
        'seed': rngseed,
        'data_path': str(root),
        'output_root': str(output_root)
    }
    with open(output_root / "params.json", "w") as f_json:
        json.dump(parameters, f_json, indent=2)

    longest_duration = 0.0

    # process each subfolder independently
    for subdir in sorted(all_subdirs):
        print(f"\n{'='*40}\nProcessing subfolder: {subdir}\n{'='*40}")

        specfml_lst, ampfml_lst = [], []
        freq_fml_lst, amp_fml_lst = [], []
        results_lst, predicted_lst = [], []

        wav_files = list(subdir.glob("*.wav"))
        mat_files = list(subdir.glob("*.mat"))
        all_files = wav_files + mat_files
        if debug_size:
            all_files = all_files[:debug_size]
        print(f"Found {len(all_files)} files ({len(wav_files)} wav, {len(mat_files)} mat) in {subdir}")

        for i, file_path in enumerate(sorted(all_files)):
            relative_path = file_path.relative_to(root)
            print(f'Processing file: {relative_path}')
            file_stem = relative_path.stem
            print(f'File {i+1}/{len(all_files)}: {file_stem}')

            output_folder = output_root / relative_path.parent
            output_folder.mkdir(exist_ok=True, parents=True)

            # unified load
            record = load_prepare_spec(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim)
            results_lst.append(record)

            denoised_spec = record["denoised_spec"]
            freqs = record["freqs"]
            times = record["times"]
            duration = record["duration"]
            sample_rate = record["sample_rate"]
            audio_data = record["audio_data"]

            # segmentation
            frame_ms = (duration / denoised_spec.shape[1]) * 1000.0
            min_dur_frames = int(round(min_dur_ms / frame_ms))
            min_gap_frames = int(round(min_gap_ms / frame_ms))
            print(f"Duration: {duration:.2f}s, Frame: {frame_ms:.2f} ms → "
                  f"min_dur: {min_dur_frames} frames, min_gap: {min_gap_frames} frames")

            segments, mask, a = getSpecSegments_smooth(
                denoised_spec,
                ampTol_hi=ampTol_hi,
                ampTol_lo=ampTol_lo,
                min_gap_frames=min_gap_frames,
                min_dur_frames=min_dur_frames,
                quantile=quantile,
                use_quantile=use_quantile
            )
            print(f'Number of segments: {len(segments)}')

            # optionally indicate segments on top rows for visualization
            denoised_vis = denoised_spec.copy()
            rows = denoised_vis.shape[0]
            n = max(0, min(front_row, rows))
            for seg in segments:
                denoised_vis[rows - n: rows, seg[0]:seg[1]] = denoised_vis.max()

            # stretch timescale (keep columns same, scale the end-time)
            duration_ext = duration * times_extended
            times_ext = np.linspace(0, duration_ext, denoised_spec.shape[1], endpoint=False)

            # synthesize frequency formulas
            subfreqfmls = synthesize_frequency(denoised_spec, times_ext, segments,
                                               time_per_seg, freqs, lyFreqlim, uyFreqlim, thickness)

            # compose final frequency formula
            fmlfreq = composeFinalFml(segments, subfreqfmls, times_ext)
            specfml_lst.append(fmlfreq)
            freq_fml = fml2latex(fmlfreq)
            freq_fml_lst.append(freq_fml)
            print(f'Frequency formula: {fmlfreq}')
            print(f'Frequency formula latex: {freq_fml}')

            # draw predicted frequency spec (relative per segment)
            t0 = time()
            freq_spec = drawRelativeSpec(segments, subfreqfmls, times_ext, denoised_spec, lyFreqlim, uyFreqlim, thickness)
            print(f"Drawing frequency took {time() - t0:.3f} s.")

            # amplitude formula (constant 1)
            fmlamp = "1"
            ampfml_lst.append(fmlamp)
            amp_fml = fml2latex(fmlamp)
            amp_fml_lst.append(amp_fml)
            print(f'Amplitude formula: {fmlamp}')
            print(f'Amplitude formula latex: {amp_fml}')

            # Save single-file formulae (YAML + MAT)
            formula_record = {
                'file': str(file_path),
                'fmlfreq': fmlfreq,
                'fmlamp': fmlamp,
                'freq_fml_latex': freq_fml,
                'amp_fml_latex': amp_fml,
            }
            formulae_subfolder = output_folder / "formulae"
            formulae_subfolder.mkdir(exist_ok=True, parents=True)

            single_formulae_yml_file = formulae_subfolder / (file_path.stem + "_formulae.yml")
            with open(single_formulae_yml_file, 'w') as f:
                yaml.dump(formula_record, f, default_flow_style=False, sort_keys=False)
            single_formulae_mat_file = formulae_subfolder / (file_path.stem + "_formulae.mat")
            sio.savemat(single_formulae_mat_file, formula_record)
            print(f"Saved formula YAML to {single_formulae_yml_file}")
            print(f"Saved formula .mat to  {single_formulae_mat_file}")

            # amplitude application (constant 1 → pass-through)
            t0 = time()
            amppredicted_spec = ampflmDraw(fmlamp, times_ext, denoised_spec, freq_spec, lyFreqlim, uyFreqlim)
            print(f"Drawing amplitude took {time() - t0:.3f} s.")

            # collect predicted record
            predicted_record = {"file": str(file_path), "predicted_spec": amppredicted_spec}
            predicted_lst.append(predicted_record)

            # plot & save (no waveform)
            output_file = output_folder / (file_path.stem + ".png")
            plotSaveResults(
                file_path, audio_data, duration_ext,
                denoised_vis, amppredicted_spec,
                longest_duration, output_file, output_folder,
                lyFreqlim, uyFreqlim, freqs,
                verbose=True,
                smooth_pred=False,   # 关闭平滑，避免线条变宽
                smooth_sigma=0.5
            )

        # ----- Save subfolder-level aggregates -----
        if len(all_files) > 0:
            rel_parent = subdir.relative_to(root)

            # MATLAB bundles
            sio.savemat(output_root / rel_parent / f"{subdir.name}_formulae.mat",
                        {"specfml_lst": specfml_lst, "ampfml_lst": ampfml_lst})
            sio.savemat(output_root / rel_parent / f"{subdir.name}_spectrogram_data.mat",
                        {"data_list": results_lst})
            sio.savemat(output_root / rel_parent / f"{subdir.name}_predicted_spec.mat",
                        {"predicted_lst": predicted_lst})

            # YAML bundle
            formulae_yaml = {
                "file": str(rel_parent),
                "specfml_lst": specfml_lst,
                "ampfml_lst": ampfml_lst,
                "freq_fml_latex": freq_fml_lst,
                "amp_fml_latex": amp_fml_lst
            }
            (output_root / rel_parent).mkdir(parents=True, exist_ok=True)
            with open(output_root / rel_parent / f"{subdir.name}_formulae.yml", 'w') as f:
                yaml.dump(formulae_yaml, f, default_flow_style=False, sort_keys=False)

            print(f"Saved subfolder bundles under {output_root / rel_parent}")

if __name__ == "__main__":
    main()

from pathlib import Path
import os, sys, librosa, json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import scipy.io as sio
from synthesizers import geneticProgram, geneticProgramAmp, fmlDrawSpec, ampflmDraw, set_seed
from random import seed
from utils import composeFinalFml, fml2latex, load_and_normalize_audio, create_audio_spectrogram, advanced_denoise_spectrogram, fmlDrawSpec_fast, ampflmDraw_fast
from scipy.ndimage import gaussian_filter
import yaml
import math
from pathlib import Path
import os, sys, librosa, json
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import scipy.io as sio
from time import time
from random import seed
from scipy.ndimage import gaussian_filter
import yaml
from fml2latex import fml2latex
import math
# compatibility for different OS
import argparse, os
# post-processing for the denoised spectrograms
from scipy.signal import savgol_filter
from scipy.ndimage import binary_closing, binary_opening, binary_dilation
sys.setrecursionlimit(20000)  # Increase limit (use cautiously)

# set a sample size for debug mode
debug_size = 0 # set 0 to turn off debug

# set random seed
# rngseed = int(sys.argv[1]) if len(sys.argv)>=2 else 47 # default: 47
# set_seed(rngseed)

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("seed", nargs="?", type=int, default=47,
#                    help="random seed (optional positional, default=47)")
#     p.add_argument("--regularizer", type=float, default=0.01,
#                    help="regularization strength")
#     p.add_argument("--output_dir", type=str, default="runs_sin_sample_test_0922",
#                    help="where to save outputs")
#     # NEW: accept dataset path from the launcher
#     p.add_argument("--data_path", type=str, default="./datasets/sin_sample_test",
#                    help="dataset folder (abs or relative path)")
#     # (optional) expose debug_size if you like
#     # p.add_argument("--debug_size", type=int, default=0)
#     return p.parse_args()

# args = parse_args()

# # ---- seeds & hyperparams ----
# rngseed = args.seed
# set_seed(rngseed)
# regularizer = args.regularizer

# # ---- paths ----
# output_dir = Path(args.output_dir).expanduser().resolve()
# output_dir.mkdir(parents=True, exist_ok=True)

# data_path = Path(args.data_path).expanduser()
# if not data_path.exists():
#     print(f"[ERROR] data_path does not exist: {data_path}", file=sys.stderr)
#     sys.exit(2)
# if not data_path.is_dir():
#     print(f"[ERROR] data_path is not a directory: {data_path}", file=sys.stderr)
#     sys.exit(2)

# # run_dir name includes seed, reg, and dataset basename
# reg_tag = str(regularizer).replace('.', 'p')
# dataset_name = data_path.name
# run_dir = output_dir / f"seed{rngseed}_reg{reg_tag}_{dataset_name}"
# run_dir.mkdir(parents=True, exist_ok=True)
# output_path = run_dir / f"plots_seed{rngseed}_reg{reg_tag}_{dataset_name}"
# output_path = str(output_path)
# # set a sample size for debug mode
# debug_size = 0 # set 0 to turn off debug

# # set random seed
rngseed = int(sys.argv[1]) if len(sys.argv)>=2 else 47 # default: 47
set_seed(rngseed)

# Constant parameters
lyFreqlim = 2000 
uyFreqlim = 8000 # 8500 for multiple-species
thickness = 8
ampTol = 0.3 # standard 0.3, binarized 0.09
amp_threshold = 0.25 # for binarized spectrograms if binarized = True
noise_threshold = 0.65 # background noise for open set it will be larger, for the closed set it will be smaller
time_per_seg = 2.5 # in synthesize-->> ga ---> while time_per_seg for evolving
front_row = 10
ampTol_hi = 0.16
ampTol_lo = 0.1
min_dur_ms = 15
min_gap_ms = 25
quantile = 0.95
estimated_K = 8 # estimated number of segments for each audio, used in segment_with_target_k
use_quantile = False
times_extended = 10
freqs_extended = 2
# set results path
data_path = "./datasets_link/Chickadee"
print(f"Data path: {data_path}")
output_path = f"1017_plots{rngseed}_Chickadee_1017_mindur{min_dur_ms}_mingap{min_gap_ms}_hysteresis{ampTol_hi}-{ampTol_lo}_combine"
# create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
print(f"Output directory: {output_path}")
# create parameter string
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
'quantile': quantile
}

def plotSaveResults(file_path, audio_data, duration, denoised_spec, predicted_spec, longest_duration, output_file, output_folder, lyFreqlim, uyFreqlim, verbose=True):
    # def plotSaveResults(file_path, audio_data, sample_rate, duration, times, freqs, denoised_spec, predicted_spec, fml, output_file, output_folder, verbose=True):
     # --- Combined Figure ---
    # expand to longest duration
    wave_times = np.linspace(0, duration, len(audio_data))
    # # 1. Expand audio_data to longest_duration
    # target_len_audio = int(math.ceil(longest_duration * sample_rate))

    # padded_audio = np.zeros(target_len_audio)
    # # padded_audio[:len(audio_data)] = audio_data  # Copy original data
    # if len(padded_audio) < len(audio_data):
    #     print(file_path, "padded_audio is too short for audio_data!")
    #     audio_data = audio_data[:len(padded_audio)]  # Truncate if too long
    #     padded_audio = np.zeros(len(audio_data))  # Reset to original length if too short
    #     padded_audio[:len(audio_data)] = audio_data  

    # else:
    #     padded_audio[:len(audio_data)] = audio_data



    # # 2. Expand denoised_spec to match longest_duration in time steps
    # # Estimate time step duration per spectrogram column
    # time_step = duration / denoised_spec.shape[1]
    # target_steps = int(longest_duration / time_step)

    # # Pad spectrogram with zeros along time axis
    # pad_width = target_steps - denoised_spec.shape[1]
    # if pad_width > 0:
    #     padded_denoised = np.pad(denoised_spec, ((0, 0), (0, pad_width)), mode='constant')
    #     padded_predicted =  np.pad(predicted_spec, ((0, 0), (0, pad_width)), mode='constant')
    # else:
    #     padded_denoised = denoised_spec
    #     padded_predicted =  predicted_spec

    # 3. Generate new time axes
    # freq_times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)
    freq_times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)
    
    fig = plt.figure(figsize=(15, 12))
    
    # Plot waveform in subplot 1
    ax1 = plt.subplot(311)
    ax1.plot(wave_times, audio_data)
    ax1.set_title("Audio Waveform")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_xlim(0, duration)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
    ax1.set_ylabel("Amplitude")
    
    # Plot denoised spectrogram in subplot 2 using specshow
    ax2 = plt.subplot(312)
    # Using specshow to display the spectrogram with time and frequency axes
    masked_denoised_spec = denoised_spec.copy()
    #masked_denoised_spec[denoised_spec < 0.15] = 0
    #masked_denoised_spec[denoised_spec >= 0.15] = 1
    image = librosa.display.specshow(masked_denoised_spec, y_coords=freqs/1000, x_coords=freq_times, x_axis='time', y_axis='hz', cmap="viridis", ax=ax2)
    ax2.set_title("Denoised Spectrogram")
    ax2.set_ylabel("Frequency (kHz)")
    ax2.set_ylim(lyFreqlim/1000, uyFreqlim/1000)
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
    plt.colorbar(image, ax=ax2, label="Magnitude")
    
    # Plot predicted spectrogram in subplot 3 using specshow
    ax3 = plt.subplot(313)
    # Optionally, apply a Gaussian filter if you wish to smooth the data
    predicted_spec_smoothed = gaussian_filter(predicted_spec, sigma=1.0)
    librosa.display.specshow(predicted_spec_smoothed, y_coords=freqs/1000, x_coords=freq_times, x_axis='time', y_axis='hz', cmap="viridis", ax=ax3)
    ax3.set_title("Predicted Spectrogram")
    ax3.set_ylabel("Frequency (kHz)")
    ax3.set_ylim(lyFreqlim/1000, uyFreqlim/1000)
    ax3.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
    plt.colorbar(image, ax=ax3, label="Magnitude")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # --- Save Separate Images if verbose ---
    if verbose:
        os.makedirs(output_folder, exist_ok=True)
        base_name = Path(file_path).stem
        
        # Waveform Figure
        fig_wave, ax = plt.subplots(figsize=(15, 4))
        plt.plot(wave_times, audio_data)
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.xlim(0, longest_duration)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
        plt.ylabel("Amplitude")
        waveform_folder = os.path.join(output_folder, "waveform")
        os.makedirs(waveform_folder, exist_ok=True)
        waveform_path = os.path.join(waveform_folder, base_name + "_waveform.png")
        plt.savefig(waveform_path, dpi=300, bbox_inches="tight")
        plt.close(fig_wave)
        
        # Denoised Spectrogram Figure with specshow
        fig_denoised = plt.figure(figsize=(15, 4))
        ax = plt.axes()
        image = librosa.display.specshow(denoised_spec, y_coords=freqs/1000, x_coords=freq_times, x_axis='time', y_axis='hz', cmap="viridis", ax=ax)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
        plt.title("Denoised Spectrogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.ylim(lyFreqlim/1000, uyFreqlim/1000)
        plt.colorbar(image, label="Magnitude")
        # plt.axis('off')
        denoised_folder = os.path.join(output_folder, "denoised")
        os.makedirs(denoised_folder, exist_ok=True)
        denoised_path = os.path.join(denoised_folder, base_name + "_denoised.png")
        plt.savefig(denoised_path, dpi=300, bbox_inches="tight")
        plt.close(fig_denoised)

        # Predicted Spectrogram Figure with specshow
        fig_predicted = plt.figure(figsize=(15, 4))
        ax = plt.axes()
        image = librosa.display.specshow(predicted_spec_smoothed, y_coords=freqs/1000, x_coords=freq_times, x_axis='time', y_axis='hz', cmap="viridis", ax=ax)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))    # Set ticks
        plt.title("Predicted Spectrogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.ylim(lyFreqlim/1000, uyFreqlim/1000)
        plt.colorbar(image, label="Magnitude")

        predicted_folder = os.path.join(output_folder, "predicted")
        os.makedirs(predicted_folder, exist_ok=True)
        predicted_path = os.path.join(predicted_folder, base_name + "_predicted.png")
        plt.savefig(predicted_path, dpi=300, bbox_inches="tight")
        plt.close(fig_predicted)

# def getSpecSegments(spec, ampTol):
#     segPairs = []
#     startT = -1
#     endT = -1
#     t = 0
#     for t in range(spec.shape[1]):
#         if (np.any(spec[:, t] > ampTol)):
#             # sound discovered
#             if (startT < 0):
#                 startT = t
#             else:
#                 endT = t
#         elif (startT >= 0 and endT >= 0):
#             # going through a segment
#             segPairs.append((startT, endT))
#             startT = -1
#             endT = -1

#     # the last segment
#     if (startT >= 0 and endT >= 0):
#         # going through a segment
#         segPairs.append((startT, endT))
#         startT = -1
#         endT = -1

#     return segPairs
# improved version of getSpecSegments with morphological operations
# to remove noise and small gaps
def _mask_to_runs(mask):
    x = mask.astype(int)
    diff1 = np.r_[x[0], np.diff(x)]
    starts = np.where(diff1 == 1)[0]
    diff2 = np.r_[np.diff(x), -x[-1]]
    ends   = np.where(diff2 == -1)[0] + 1
    return starts, ends

# improved version of getSpecSegments with morphological operations
def getSpecSegments_smooth(
    spec,
    ampTol_hi,
    ampTol_lo=None,       # default to 0.6*hi
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

    # run-length edits
    s, e = _mask_to_runs(mask)
    for i in range(len(s)):
        if (e[i] - s[i]) < int(min_dur_frames):
            mask[s[i]:e[i]] = False

    s, e = _mask_to_runs(mask)
    for i in range(len(e) - 1):
        gap = s[i+1] - e[i]
        if gap <= int(min_gap_frames):
            mask[e[i]:s[i+1]] = True

    starts, ends = _mask_to_runs(mask)
    segPairs = list(zip(starts, ends))
    print("Number of Segment Pairs:", len(segPairs))
    return segPairs, mask, a

def load_prepare_spec(file_path, noise_threshold, amp_threshold):
    # first load the wave file
    sample_rate, audio_data = load_and_normalize_audio(file_path)

    # create initial spectrogram
    original_spec, _ = create_audio_spectrogram(audio_data, sample_rate)

    # apply advanced denoising
    denoised_spec = advanced_denoise_spectrogram(original_spec, noise_threshold, amp_threshold, normalize_time = False, binarize = False)    # Enhance contrast
    # Load and process audio
    duration = len(audio_data) / sample_rate # total samples / sample rate = total duration in seconds

    # return
    return denoised_spec, duration, sample_rate, audio_data


def cut_spectrogram(denoised_spec, freqs, lyFreqlim, uyFreqlim):
    lyFreqind = 0
    for i in range(len(freqs)):
        if (freqs[i] >= lyFreqlim):
            lyFreqind = i
            break

    uyFreqind = freqs[-1]
    for i in range(len(freqs) - 1, -1, -1):
        if (freqs[i] <= uyFreqlim):
            uyFreqind = i
            break

    denoised_spec = denoised_spec[lyFreqind:uyFreqind + 1, :]
    freqs = freqs[lyFreqind:uyFreqind + 1]
    return denoised_spec, freqs

#  process spectrograms
def process_spectrogram(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim):

    # Load the spectrogram and related audio metadata
    denoised_spec, duration, sample_rate, audio_data = load_prepare_spec(file_path, noise_threshold, amp_threshold)
    print(f'Processing file: {file_path}')

    # Define time and frequency axes
    # freqs = np.linspace(lyFreqlim, uyFreqlim, denoised_spec.shape[0])  # Linear frequency scale
    freqs = np.linspace(0, sample_rate / 2, denoised_spec.shape[0]) 
    times = np.linspace(0, duration, denoised_spec.shape[1])  # Time axis
    
    # Extract relevant frequency range
    denoised_spec, freqs = cut_spectrogram(denoised_spec, freqs, lyFreqlim, uyFreqlim)
    average_freq = np.mean(freqs)
    
    # Save processed data as a dictionary
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

def save_predicted_spec(file_path, predicted_spec):
    predicted_record = {
        "file": str(file_path),
        "predicted_spec": predicted_spec
    }
    return predicted_record

def synthesize_frequency(denoised_spec, times, segments, time_per_seg, freqs, lyFreqlim, uyFreqlim, thickness = 5):
    # synthesize a subformula for each segment
    subfreqfmls = []
    for seg in segments:
        spec_seg = denoised_spec[:, seg[0]:seg[1]]
        #time_seg = times[seg[0]:seg[1]]
        time_seg = times[0:seg[1] - seg[0]]
        # print(f"current segment: {seg}")
        # print(f"time_seg: {time_seg}")
        average_freq = np.mean(freqs)
        fmlfreq, _ = geneticProgram(time_seg, spec_seg, time_per_seg, thickness, lyFreqlim, uyFreqlim, fmlSeed = "")
        subfreqfmls.append(fmlfreq)
 
    # sanity check
    assert(len(subfreqfmls) == len(segments))
    return subfreqfmls


def drawRelativeSpec(segments, subfmls, times, denoised_spec, lyFreqlim, uyFreqlim, thickness):
    assert len(subfmls) == len(segments), "number of subfmls does not match number of segments"
    pred_spec = np.zeros(denoised_spec.shape)
    for segI, seg in enumerate(segments):
        time_seg = times[0:seg[1] - seg[0]]
        freq_spec = fmlDrawSpec(subfmls[segI], time_seg, denoised_spec[:,seg[0]:seg[1]], lyFreqlim, uyFreqlim, thickness)
        pred_spec[:, seg[0]:seg[1]] = freq_spec
    return pred_spec


def synthesize_amplitude(denoised_spec, times, freq_spec, time_per_seg, freqs, lyFreqlim, uyFreqlim):
    # synthesize amplitude formula
    subampfmls = []
    for seg in segments:
        spec_seg = denoised_spec[:, seg[0]:seg[1]]
        time_seg = times[seg[0]:seg[1]]
        freq_spec_seg = freq_spec[:, seg[0]:seg[1]]
        # print(f"current segment: {seg}")
        average_freq = np.mean(freqs)
        fmlamp, _ = geneticProgramAmp(time_seg, spec_seg, freq_spec_seg, time_per_seg, lyFreqlim, uyFreqlim, fmlSeed = "")
        subampfmls.append(fmlamp)

    # sanity check
    assert(len(subampfmls) == len(segments))
    return subampfmls


if __name__ == "__main__":

    # Read in the dataset
    output_root = Path(output_path)  # Using Path for better cross-platform compatibility
    output_root.mkdir(exist_ok = True)     # Creates directory if it doesn't exist, won't raise error if it does
    root = Path(data_path) # set the base directory for the dataset

    # Get all unique directories containing .wav files
    all_subdirs = set()
    for file_path in root.glob("**/*.wav"):
        all_subdirs.add(file_path.parent)
    
    '''
    for subdir in all_subdirs:
        wav_files = list(subdir.glob("*.wav"))
        if debug_size:
            wav_files = wav_files[:debug_size]
        for file_path in wav_files:
            duration = librosa.get_duration(filename = file_path)
    '''
            # try:
            #     duration = librosa.get_duration(path = file_path)
            #     if duration > longest_duration:
            #         longest_duration = duration
            #         longest_file = file_path.relative_to(root)
            # except Exception as e:
            #     print(f"Error processing {file_path.relative_to(root)}: {e}")

    # print(f"\n Longest audio duration: {longest_duration:.2f} seconds")
    # add longest duration to parameters dict
    # parameters["longest_duration"] = longest_duration
    # write parameters into json
    # with open(f"./{output_path}/para.json", "w") as f_json:
    #     json.dump(parameters, f_json, indent=4)
        
    # if longest_file:
    #     print(f" File: {longest_file}")

    # Find longest duration
    longest_duration = 0.0
    longest_file = None

    # Process each subfolder separately
    for subdir in all_subdirs:
        print(f"\n{'='*40}\nProcessing subfolder: {subdir}\n{'='*40}")
        
        # Initialize lists for this subfolder only
        specfml_lst = []
        ampfml_lst = []
        freq_fml_lst = []  # Save latex expressions for all frequency formulae
        amp_fml_lst = []   # Save latex expressions for all amplitude formulae
        freq_fml_readable_lst = []  # Save human-readable frequency formulae
        amp_fml_readable_lst = []   # Save human-readable amplitude formulae
        results_lst = []
        predicted_lst = []


        # Get all wav files in this specific subfolder (non-recursive)
        wav_files = list(subdir.glob("*.wav"))
        if debug_size:
            wav_files = wav_files[:debug_size]
        print(f"Found {len(wav_files)} .wav files in {subdir}")
        
        for i, file_path in enumerate(wav_files):
            # Create the relative path
            relative_path = file_path.relative_to(root)
            print(f'Processing file: {relative_path}')
            # leave the file name and remove its extension and partent path
            file_name = relative_path.stem  # Get the file name without extension
            # file_name = file_path.name
            print(f'File {i+1}/{len(wav_files)}: {file_name}')
            
            # Create the subfolder structure in the output folder
            output_folder = output_root / relative_path.parent
            output_folder.mkdir(exist_ok=True, parents=True)
            
            # Process the spectrogram
            record = process_spectrogram(file_path, noise_threshold, amp_threshold, lyFreqlim, uyFreqlim)
            
            results_lst.append(record)
            
            file_name = record["file"]
            denoised_spec = record["denoised_spec"]
            freqs = record["freqs"]
            times = record["times"]
            duration = record["duration"]
            sample_rate = record["sample_rate"]
            audio_data = record["audio_data"]
            average_freq = record["average_freq"]

            # Synthesis
            # Part spectrogram into segments
            # segments = getSpecSegments(denoised_spec, ampTol)
            # duration / denoised_spec.shape[1] = time per frame in seconds
            frame_ms = (duration / denoised_spec.shape[1]) * 1000.0
            min_dur_frames = int(round(min_dur_ms / frame_ms))
            min_gap_frames = int(round(min_gap_ms / frame_ms))
            print(f"Duration: {duration:.2f}, Frame duration: {frame_ms:.2f} ms")
            print(f"Min_dur_ms: {min_dur_ms}, Min segment length: {min_dur_frames} frames|| Min_gap_ms: {min_gap_ms}, Min gap length: {min_gap_frames} frames")
            
            segments, mask, a = getSpecSegments_smooth(
                denoised_spec,
                ampTol_hi=ampTol_hi,
                ampTol_lo=ampTol_lo,       # if None → 0.6 * ampTol_hi
                min_gap_frames=min_gap_frames,    # merge gaps ≤ this (frames)
                min_dur_frames=min_dur_frames,    # drop segments < this (frames)
                quantile=quantile,        # per-column aggregation quantile
                use_quantile=use_quantile
                )
            
            print(f'Number of segments: {len(segments)}')
            # print(f'Segments: {segments}')

            # for each segment, make it 10 times longer and then synthesize
            # extended_segments = []
            # for seg in segments:
            #     seg_length = seg[1] - seg[0]
            #     extended_length = int(seg_length * times_extended)  # Extend to 10 times the original length
            #     new_end = min(seg[0] + extended_length, denoised_spec.shape[1])  # Ensure we don't exceed bounds
            #     extended_segments.append((seg[0], new_end))
            # segments = extended_segments
            # print(f'Extended segments for synthesis: {segments}')

            # for each spectrogram, duration and times are 10 times longer
            duration *= times_extended
            times = np.linspace(0, duration, denoised_spec.shape[1], endpoint=False)

            # Synthesize the frequencies
            subfreqfmls = synthesize_frequency(denoised_spec, times, segments, time_per_seg, freqs, lyFreqlim, uyFreqlim, thickness)

            # Compose the final formula
            fmlfreq = composeFinalFml(segments, subfreqfmls, times)

            print(subfreqfmls)


            specfml_lst.append(fmlfreq)
            # Frequency latex
            freq_fml = fml2latex(fmlfreq)
            freq_fml_lst.append(freq_fml)

            print(f'Frequency formula: {fmlfreq}')
            print(f'Frequency formula latex: {freq_fml}')
            
            # Plot predicted spectrogram
            startT = time()
            #freq_spec = fmlDrawSpec(fmlfreq, times, denoised_spec, lyFreqlim, uyFreqlim, thickness)
            freq_spec = drawRelativeSpec(segments, subfreqfmls, times, denoised_spec, lyFreqlim, uyFreqlim, thickness)
            print(f"Drawing frequency took {time() - startT} secs.")

            # Synthesize formulae for amplitude
            '''
            subampfmls = synthesize_amplitude(denoised_spec, times, freq_spec, time_per_seg, freqs, lyFreqlim, uyFreqlim)
            
            # Compose the final formula
            fmlamp = composeFinalFml(segments, subampfmls, times)
            '''
            fmlamp = "1"
            ampfml_lst.append(fmlamp)
            
            # Amplitude latex
            amp_fml = fml2latex(fmlamp)
            amp_fml_lst.append(amp_fml)
            # amp_fml_read = fml2readable(fmlamp)
            # amp_fml_readable_lst.append(amp_fml_read)
            print(f'Amplitude formula: {fmlamp}')
            print(f'Amplitude formula latex: {amp_fml}')

            ## Save single file formulae as YAML
            formula_record = {
                'file': str(file_path),
                'fmlfreq': fmlfreq,
                'fmlamp': fmlamp,
                'freq_fml_latex': freq_fml,
                'amp_fml_latex': amp_fml,
                # 'freq_fml_readable': freq_fml_read,
                # 'amp_fml_readable': amp_fml_read

            }
            # Create formulae subfolder under current output folder
            formulae_subfolder = output_folder / "formulae"
            formulae_subfolder.mkdir(exist_ok=True, parents=True)

            # Save YAML into formulae subfolder
            single_formulae_yml_file = formulae_subfolder / (file_path.stem + "_formulae.yml")
            print(f"Saved formula YAML to {single_formulae_yml_file}")
            # Save the formulae for this file
            with open(single_formulae_yml_file, 'w') as f:
                yaml.dump(formula_record ,
                 f, default_flow_style=False, sort_keys=False)
            # save as .mat file
            single_formulae_mat_file = formulae_subfolder / (file_path.stem + "_formulae.mat")
            sio.savemat(single_formulae_mat_file, formula_record)
            print(f"Saved formula .mat to {single_formulae_mat_file}")

            # Plot predicted spectrogram
            startT = time()
            amppredicted_spec = ampflmDraw(fmlamp, times, denoised_spec, freq_spec,lyFreqlim, uyFreqlim)
            print(f"Drawing amplitude took {time() - startT} secs.")
          
            # predicted_spec[file_path.stem] = amppredicted_spec
            # Before processing wav files, add an index counter
             # Process the spectrogram
            predicted_record = save_predicted_spec(file_path, amppredicted_spec)
            predicted_lst.append(predicted_record)
            
            file_name = predicted_record["file"]
            predicted_spec = predicted_record["predicted_spec"]
            
            # Display segments on denoised spectrogram
            for seg in segments:
                denoised_spec[0:front_row, seg[0]:seg[1]] = 1
            
            # Process and save results indise corresponding subfolders
            output_file = output_folder / (file_path.stem + ".png")  # Using stem to get filename without extension
            
            # Plot and save the reconstructed spectrograms
            plotSaveResults(file_path, audio_data, duration, denoised_spec, predicted_spec, longest_duration, output_file, output_folder, lyFreqlim, uyFreqlim, verbose=True)
            
            
        # Only save the subfolder's results once processing of all files in the subfolder is complete
        if wav_files:  # Only save if there were files processed
            # Save formulae for this subfolder
            subfolder_name = subdir.name
            formulae_file = output_root / relative_path.parent / f"{subfolder_name}_formulae.mat"
            sio.savemat(formulae_file, {"specfml_lst": specfml_lst, "ampfml_lst": ampfml_lst})
            print(f"Saved formulae to {formulae_file}")

            # Save spectrogram data for this subfolder
            save_path = output_root / relative_path.parent / f"{subfolder_name}_spectrogram_data.mat"
            sio.savemat(str(save_path), {"data_list": results_lst})
            print(f"Saved spectrogram data to {save_path}")

            # Save prediced_spec for this subfolder
            predicted_spec_file = output_root / relative_path.parent / f"{subfolder_name}_predicted_spec.mat"
            sio.savemat(str(predicted_spec_file), {"predicted_lst": predicted_lst})
            print(f"Saved predicted spectrogram to {predicted_spec_file}")
            
            # save as yml files
            # Save formulae for this subfolder
            formulae_file = output_root / relative_path.parent / f"{subfolder_name}_formulae.yml"
            formulae = {
                "file": str(relative_path.parent),
                "specfml_lst": specfml_lst, 
                "ampfml_lst": ampfml_lst,
                "freq_fml_latex": freq_fml_lst,
                "amp_fml_latex": amp_fml_lst
                # "freq_fml_readable": freq_fml_readable_lst,
                # "amp_fml_readable": amp_fml_readable_lst
            }
            with open(formulae_file, 'w') as file:
                yaml.dump(formulae, file, default_flow_style=False, sort_keys=False)
            print(f"Saved formulae to {formulae_file}")

            # # Save spectrogram data for this subfolder
            # save_path = output_root / relative_path.parent / f"{subfolder_name}_spectrogram_data.mat"
            # result = {"data_list": results_lst}
            # with open(save_path, 'w') as file:
            #     yaml.dump(result, file, default_flow_style=False, sort_keys=False)
            # print(f"Saved spectrogram data to {save_path}")

            # # Save prediced_spec for this subfolder
            # predicted_spec_file = output_root / relative_path.parent / f"{subfolder_name}_predicted_spec.mat"
            # predicted_spec_data = {"predicted_lst": predicted_lst}
            # with open(predicted_spec_file, 'w') as file:
            #     yaml.dump(predicted_spec_data, file, default_flow_style=False, sort_keys=False)
            # print(f"Saved predicted spectrogram to {predicted_spec_file}")

# this is the utility functions for formula generation and evaluation: the difference between the old one is without average frequency in the fml
# imports
from grammar import V, T, U, B, X, XB, I
from funcWrappers import wrapExtGetHeadAndInputs, wrapExtParseEvalBatch, wrapExtParseEvalTwoVarBatch
from scipy.io import wavfile
import numpy as np
import random, librosa
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
import scipy.io as sio
from decompose import decomp_fml_string
from math import isinf, isnan, floor
from collections import Counter
import torch
import re
import os

# computes the majority class ratio
def mca(num_lst):
    _, count = Counter(num_lst).most_common(1)[0]
    return float(count) / len(num_lst)


# a quick compute of literal accuracy of the predicted RPN compared to the target RPN
def model_acc(pred, target, ignore_index):
    padding_mask = (target == ignore_index)
    correct_bool = torch.eq(torch.argmax(pred, dim = -1), target)
    correct_bool = torch.logical_and(correct_bool, torch.logical_not(padding_mask))
    nb_correct = torch.sum(correct_bool)
    return nb_correct / torch.sum(torch.logical_not(padding_mask))


def _inds_to_onehot(inds_mat, full_vocab, device = "cpu"):
    # converts a batch of indices to a batch of one hot vectors
    # the main loop
    onehots = np.zeros((inds_mat.shape[0], inds_mat.shape[1], len(full_vocab)))
    for rowI in range(inds_mat.shape[0]):
        inds = inds_mat[rowI, :]
        for indI in range(len(inds)):
            onehot = np.zeros((len(full_vocab),))
            onehot[int(inds[indI])] = 1
            onehots[rowI, indI, :] = onehot
    return torch.tensor(onehots, dtype = torch.int64).to(device)


def _rpn_to_inds_pad(rpn, full_vocab, max_fml_len = 30):
    # first, construct the entire formula string
    full_str = "<bos> " + rpn
    rpn_len = len(rpn.split())

    # Sanity check
    assert rpn_len <= max_fml_len, "RPN size is larger than maximum allowed. What?"

    # <pad> and <end>, if needed
    if (rpn_len < max_fml_len):
        for padI in range(max_fml_len - rpn_len):
            full_str += " <pad>"
    full_str += " <eos>"

    # now, convert the entire formula string to a numpy array of indices
    tok2ind_dict = {tok : idx for idx, tok in enumerate(full_vocab)}
    full_str_tok_lst = full_str.split()
    indices = np.array([tok2ind_dict[tok] for tok in full_str_tok_lst])

    # sanity check
    assert len(indices) == max_fml_len + 2, "Indices length is not the same as the maximum formula length."

    # return
    return indices


def inference(model, inputs, targets, full_vocab, max_fml_len, device = "cpu", batch_size = 1024):
    ## use the model to do inference on the provided inputs
    # number of tokens as prompts from targets
    num_target_prompts = 0

    # get the index for the <bos> as initial prompt
    bos_index = full_vocab.index("<bos>")

    # batch loop
    startI = 0
    prompt = None
    model.eval()
    with torch.no_grad():
        while (startI < inputs.shape[0]):
            # generate the initial prompt
            cur_prompt = None
            cur_inputs = None
            if (batch_size < inputs.shape[0] - startI):
                cur_prompt = torch.full((batch_size, 1), bos_index, dtype = torch.int64).to(device)
                cur_inputs = inputs[startI:startI + batch_size, :].to(device)

                # add the target prompt
                if (num_target_prompts):
                    cur_prompt[:, :num_target_prompts] = targets[startI:startI + batch_size, :num_target_prompts]
            else:
                cur_prompt = torch.full((inputs.shape[0] - startI, 1), bos_index, dtype = torch.int64).to(device)
                cur_inputs = inputs[startI:, :].to(device)

                # add the target prompt
                if (num_target_prompts):
                    cur_prompt[:, :num_target_prompts] = targets[startI:, :num_target_prompts]

            # inference loop
            for tokI in range(num_target_prompts, max_fml_len):
                # generate the next prediction
                pred_next = model(cur_inputs, cur_prompt)

                # convert the predicted vectors into token indices
                next_tok = pred_next[:, -1, :].argmax(dim = -1).view(-1, 1)
                cur_prompt = torch.cat((cur_prompt, next_tok), dim = -1)

            # increment startI
            startI += batch_size

            # concatentate the entire prompt
            if (prompt is None):
                prompt = cur_prompt
            else:
                prompt = torch.cat((prompt, cur_prompt), dim = 0)
    return prompt, pred_next

def set_seed(seed = 47):
    '''Completely set seeds'''
    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional for DataLoader workers
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2

    
def idx_to_toks(idx, full_vocab, num_fml_per_set = 1):
    # Used to convert a predicted index array back to the formula used for evaluation
    toks = []
    rpns = []
    if (len(idx.shape) == 2): # batch data
        for fmlI in range(idx.shape[0]):
            fml_toks = []
            for tokI in range(idx.shape[1]):
                fml_toks.append(full_vocab[idx[fmlI, tokI]])

            # first, strip the fml_toks from the front
            while (fml_toks and ("<" in fml_toks[0] and ">" in fml_toks[0])):
                fml_toks = fml_toks[1:]

            # construct the RPN formula set
            cur_fml_lst = []
            for tok in fml_toks:
                if (len(cur_fml_lst) != 0 and ("<" in tok and ">" in tok)):
                    break
                elif ("<" not in tok and ">" not in tok):
                    cur_fml_lst.append(tok)
            rpns.append(" ".join(cur_fml_lst))

            # append the indices and the tokens
            toks.append(fml_toks)
    else: # single data
        fml_toks = []
        for tokI in range(idx.shape[0]):
            fml_toks.append(full_vocab[idx[tokI]])

        # first, strip the fml_toks from the front
        while (fml_toks and ("<" in fml_toks[0] and ">" in fml_toks[0])):
            fml_toks = fml_toks[1:]

        for tok in fml_toks:
            if ("<" not in tok and ">" not in tok):
                toks.append(tok)
            else:
                break
        rpns = " ".join(toks)

    # return
    return  (toks, rpns)


# Utility function
def list_flatten(lst):
    # base case
    if (len(lst) == 0):
        return lst

    # inductive cases
    if (isinstance(lst[0], list)):
        res = list_flatten(lst[0])
        return res + list_flatten(lst[1:])
    else:
        return [lst[0]] + list_flatten(lst[1:])


# tells if a string contains a number of various formats
def is_number(text) -> bool:
    return bool(re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", text))


def lispToRPN(lispStr) -> str:
    # get rid of the parenthesis
    return pnToRPN(lispStr.replace("(", "").replace(")", ""))


def isfloat(numStr):
    try:
        float(numStr)
        return True
    except ValueError:
        return False


def pnToRPN(pn) -> str:
    # assuming the pn is valid
    pn_lst = pn.split()
    pn_lst.reverse()
    stack = []

    # loop through each token in the RPN
    try:
        for tok in pn_lst:
            if (tok in V or isfloat(tok)):
                stack.append(tok)
            elif (tok in U + ["++", "--", "round"]):
                a = stack.pop()
                stack.append(" ".join([a, tok]))
            elif (tok in ["+", "-", "*", "/", "min_2", "max_2", "argmax_2", "argmin_2"]):
                a = stack.pop()
                b = stack.pop()
                stack.append(" ".join([a, b, tok]))
            elif (tok in ["min_3", "max_3", "argmax_3", "argmin_3"]):
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                stack.append(" ".join([a, b, c, tok]))
            elif (tok in ["min_4", "max_4", "argmax_4", "argmin_4"]):
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                d = stack.pop()
                stack.append(" ".join([a, b, c, d, tok]))
            elif (tok in ["min_5", "max_5", "argmax_5", "argmin_5"]):
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                d = stack.pop()
                e = stack.pop()
                stack.append(" ".join([a, b, c, d, e, tok]))
            else:
                print(f"{tok} does not belong to the grammar.")
                exit()

        rpn = stack.pop()
        assert len(stack) == 0, "stack is not empty"
        return rpn
    except Exception as e:
        # conversion failed
        return None

# Draw amplitude with long formula but much faster
def ampflmDraw_fast(fml, times, spec, freq_spec):
    """
    Draw amplitude prediction on spectrogram using the given formula.
    
    Args:
        fml: formula string
        times: 1D numpy array of original time points (may differ in length from spec columns)
        spec: 2D numpy array spectrogram (freq bins x time bins)
        freq_spec: 2D numpy array frequency spec (freq bins x time bins), aligned with spec
        average_freq: float, average frequency for formula evaluation
    
    Returns:
        amppredicted_spec: 2D numpy array amplitude predictions (same shape as spec)
    """
    # preprocess the long formula, assuming fml is of format "(cond (& ...) ...)"
    time_segment_pairs, subfmls = decomp_fml_string(fml)
    
    # Pre-normalize the input spectrogram amplitude to [0.1, 1]
    spec_min = np.min(spec)
    spec_max = np.max(spec)
    spec_norm = (spec - spec_min) / (spec_max - spec_min)  # normalize [0,1]
    spec_norm = spec_norm * 0.9 + 0.1  # scale to [0.1,1]

    amppredicted_spec = np.zeros(spec.shape)
    height, width = spec.shape

    # Normalize frequencies - adjust upper bound as needed
    normed_freqs = np.linspace(0, 2, height)

    # Create spec_times aligned with freq_spec width
    spec_times = np.linspace(times[0], times[-1], freq_spec.shape[1])

    for segI in range(len(time_segment_pairs)):
        seg_start, seg_end = time_segment_pairs[segI]

        # Use spec_times to create mask matching freq_spec columns
        seg_mask = (spec_times >= seg_start) & (spec_times < seg_end)
        subtimes = spec_times[seg_mask]

        if len(subtimes) == 0:
            continue

        freq_spec_segment = freq_spec[:, seg_mask]

        # Evaluate predicted amplitudes on this segment
        pred_amps = wrapExtParseEvalTwoVarBatch(subfmls[segI], normed_freqs, subtimes, freq_spec_segment)

        for f in range(height):
            for t in range(len(subtimes)):
                # Skip if no energy at this freq/time bin
                if freq_spec_segment[f, t] == 0:
                    continue

                pred_amp = pred_amps[f, t]

                # Handle NaN/Inf
                if np.isnan(pred_amp):
                    pred_amp = 0.1
                elif np.isinf(pred_amp):
                    pred_amp = 1.0
                else:
                    pred_amp = max(min(pred_amp, 1.0), 0.1)

                # Map predicted amplitude to normalized scale
                pred_amp_norm = (pred_amp - 0.1) / 0.9

                # Find the column index in full spec to write to
                col_idx = np.where(seg_mask)[0][t]

                amppredicted_spec[f, col_idx] = pred_amp_norm

    return amppredicted_spec

# Draw spectrogram with long formula but much faster
def fmlDrawSpec_fast(fml, times, spec, thickness=5.0):
    """
    Draw predicted frequencies onto a spectrogram grid.
    
    Args:
        fml: String formula for frequency prediction.
        times: 1D array of time points for prediction (may differ from spec width).
        spec: 2D numpy array (freq bins x time bins).
        average_freq: Average frequency for your formula.
        thickness: Line thickness in pixels.
        
    Returns:
        freq_spec: 2D numpy array with drawn prediction.
    """
    time_segment_pairs, subfmls = decomp_fml_string(fml)

    height, width = spec.shape
    freq_spec = np.zeros_like(spec)
    draw_val = 0.1

    # Normalize frequencies (0 to 1 for mapping)
    normed_freqs = np.linspace(0, 1, height)

    # Create spec_times aligned with spectrogram columns
    spec_times = np.linspace(times[0], times[-1], width)

    for segI in range(len(time_segment_pairs)):
        seg_start, seg_end = time_segment_pairs[segI]

        seg_mask = (spec_times >= seg_start) & (spec_times < seg_end)
        subtimes = spec_times[seg_mask]
        if len(subtimes) == 0:
            continue

        # Evaluate predicted frequencies for this segment using only time vector
        pred_freqs = wrapExtParseEvalBatch(subfmls[segI], subtimes)
        assert len(pred_freqs) == len(subtimes)

        for t_idx, pred_freq in enumerate(pred_freqs):
            # Map predicted frequency (0-1) to nearest freq bin
            freq_idx = np.abs(normed_freqs - pred_freq).argmin()

            # Clamp freq_idx to valid range
            freq_idx = max(0, min(height - 1, freq_idx))

            # Map time to pixel coordinate in spectrogram
            x_pos = np.interp(subtimes[t_idx], spec_times, np.arange(width))
            x_pixel = int(round(x_pos))

            # Draw with thickness (square around center)
            half_thick = int(thickness // 2)
            for e_x in range(-half_thick, half_thick + 1):
                for e_y in range(-half_thick, half_thick + 1):
                    x = x_pixel + e_x
                    y = freq_idx + e_y
                    if 0 <= x < width and 0 <= y < height:
                        freq_spec[y, x] = draw_val

    return freq_spec

# the helper functions
def composeFinalFml(segments, subfmls, times):
    # Sanity check
    assert(len(segments) == len(subfmls))
    
    # the main loop
    finalfml = "(cond"
    for segI in range(len(segments)):
        seg = segments[segI]
        fml = subfmls[segI]
        # finalfml = f"{finalfml} ((&& (>= t {times[seg[0]]}) (<= t {times[seg[1]]})) {fml})"
        end_idx_inclusive = min(seg[1], len(times) - 1)
        finalfml = (
            f"{finalfml} ((&& (>= t {times[seg[0]]:.3f}) (<= t {times[end_idx_inclusive]:.3f})) {fml})"
        )
    finalfml = f"{finalfml} (else 0))"
    return finalfml


# function to convert formula in lisp notation to latex
def fml2latex(fml):
    # get the first and rest of the formula
    [head, inputs] = wrapExtGetHeadAndInputs(fml)
    # remove blank string from inputs
    inputs = [inp for inp in inputs if inp != ""]
    # base case: terminal
    if (len(inputs) == 0): return head


    # inductive case
    if (head in U):
        # sanity check
        assert(len(inputs) == 1)
        match head:
            case "sqr": return f"{fml2latex(inputs[0])}^2"
            case "sqrt": return f"\\sqrt{{{fml2latex(inputs[0])}}}"
            case "exp": return f"\\exp{{{fml2latex(inputs[0])}}}"
            case "neg": return f"\\left(-{fml2latex(inputs[0])}\\right)"
            case "sin": return f"\\sin\\left({fml2latex(inputs[0])}\\right)"
            case "tan": return f"\\tan\\left({fml2latex(inputs[0])}\\right)"
            case "cos": return f"\\cos\\left({fml2latex(inputs[0])}\\right)"
            case "log": return f"\\log\\left({fml2latex(inputs[0])}\\right)"
            case "abs": return f"\\left|{fml2latex(inputs[0])}\\right|"
            case _: print(f"Unary operator not supported: {head}")
    elif (head in (B + X + XB)):
        assert(len(inputs) == 2)
        match head:
            case "+": return f"{fml2latex(inputs[0])} + {fml2latex(inputs[1])}"
            case "-": return f"{fml2latex(inputs[0])} - {fml2latex(inputs[1])}"
            case "*": return f"{fml2latex(inputs[0])} \\times {fml2latex(inputs[1])}"
            case "/": return f"\\frac{{{fml2latex(inputs[0])}}}{{{fml2latex(inputs[1])}}}"
            case "==": return f"{fml2latex(inputs[0])} = {fml2latex(inputs[1])}"
            case ">": return f"{fml2latex(inputs[0])} > {fml2latex(inputs[1])}"
            case "<": return f"{fml2latex(inputs[0])} < {fml2latex(inputs[1])}"
            case ">=": return f"{fml2latex(inputs[0])} \\ge {fml2latex(inputs[1])}"
            case "<=": return f"{fml2latex(inputs[0])} \\le {fml2latex(inputs[1])}"
            case "&&": return f"{fml2latex(inputs[0])} \\land {fml2latex(inputs[1])}"
            case "||": return f"{fml2latex(inputs[0])} \\lor {fml2latex(inputs[1])}"
            case _: print(f"Binary operator not supported: {head}")
    elif (head in I):
        assert(len(inputs) == 3)
        return f"\\begin{{cases}} {fml2latex(inputs[1])} & \\text{{if }} {fml2latex(inputs[0])} \\\\ {fml2latex(inputs[2])} & \\text{{otherwise}}\\end{{cases}}"
    elif (head == "cond"):
        latex = f"\\begin{{cases}}"
        for inI in range(len(inputs) - 1):
            [subhead, subinputs] = wrapExtGetHeadAndInputs(inputs[inI])
            latex = f"{latex} {fml2latex(subinputs[0])} & \\text{{if }} {fml2latex(subhead)} \\\\"
        [_, subinputs] = wrapExtGetHeadAndInputs(inputs[inI + 1])
        latex = f"{latex} {fml2latex(subinputs[0])} & \\text{{otherwise}} \\\\"
        latex = f"{latex} \\end{{cases}}"
        return latex

def load_and_normalize_audio(file_path):
    target_sr = 44100
    audio_data, sample_rate = librosa.load(file_path, sr=target_sr)  # All files resampled to 44100 Hz
    # sample_rate, audio_data = wavfile.read(file_path)
    # print(sample_rate) # 44100
    audio_data = audio_data.astype(float)
    if len(audio_data.shape) > 1:
        #! audio_data = audio_data.mean(axis = 1)
        audio_data = audio_data.sum(axis = 1)
    audio_data = audio_data / np.max(np.abs(audio_data))
    # Normalize to [0,1]
    # audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    return sample_rate, audio_data

# pre FFT processing
def bandpass_filter(signal, fs):
    nyquist = 0.5 * fs
    lowcut = 1000
    highcut = 9000
    order=5
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def create_audio_spectrogram(audio_data, sample_rate, window_size_ms=5):
    # Filter the signal in the time domain
    filtered_audio = bandpass_filter(audio_data, sample_rate)
    
    
    # time window size and hop size
    window_size = int(sample_rate * window_size_ms / 1000)
    hop_size = window_size // 4  # 75% overlap
    window = np.hanning(window_size)
    n_windows = (len(filtered_audio) - window_size) // hop_size + 1
    # frequency buckets
    window_size_ms_freq = 30
    n_fft = int(sample_rate * 3 * window_size_ms_freq / 1000)
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate) / 1000  # Convert to kHz
    spectrogram = np.zeros((len(freqs), n_windows))
    print(f'spectrograms shape: {spectrogram.shape}') # (1985, 1198)
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        segment = filtered_audio[start:end] * window
        fft_result = np.fft.rfft(segment, n=n_fft)
        spectrogram[:, i] = np.abs(fft_result)
    return spectrogram, freqs



# post FFT processing
# def create_audio_spectrogram(audio_data, sample_rate, window_size_ms = 30):
#     """
#     Create spectrogram from audio data using raw magnitudes
#     """
#     # start with time window size
#     # STFT30ms window with 75% overlap
#     # 1323 ----> 44100Hz * 30ms / 1000 = 1323 samples
#     # print(sample_rate) # 48000
#     window_size = int(sample_rate * window_size_ms / 1000)
#     # Sets overlap between consecutive windows (75% overlap)
#     hop_size = window_size // 4
#     window = np.hanning(window_size) # smooth the signal
#     # Number of windows we can compute over the audio signal
#     n_windows = (len(audio_data) - window_size) // hop_size + 1

#     # start with frequency buckets
#     n_fft  = window_size
#     buc_size = n_fft // 2 + 1
#     # print(f'n_fft: {n_fft}') # 1440
#     # print(f'buc_size: {buc_size}') # 721
#     freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
#     # print(f'freqs shape: {freqs.shape}') # 721
#     low_freq = 1000
#     high_freq = 6000
#     # Identify indices corresponding to frequencies between low_freq and high_freq
#     valid_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    
#     # Initialize spectrogram array using only the selected frequency bins
#     spectrogram = np.zeros((len(valid_idx), n_windows))
#     # Compute the spectrogram
#     # spectrogram = np.zeros((buc_size, n_windows)) 
#     print(f'spectrogram shape: {spectrogram.shape}') # (721, 262)
#     for i in range(n_windows):
#     # for j in range(buc_size):
#         # Extract the current window
#         start = i * hop_size
#         end = start + window_size
#         # Apply the Hanning window to the segment
#         segment = audio_data[start:end] * window
#         windowed = segment * window
#         # Compute the FFT
#         fft = np.abs(np.fft.rfft(windowed))
#         # print(f'fft shape: {fft.shape}') # 1440
#         # print(f'valid idx shape: {valid_idx.shape}')
#         spectrogram[:, i] = fft[valid_idx]

#     filtered_freqs = freqs[valid_idx]
#     # More aggressive power normalization
#     spectrogram = np.power(spectrogram, 0.5)
#     spectrogram /= np.max(spectrogram) + 1e-10
#     return filtered_freqs, spectrogram

def advanced_denoise_spectrogram(spectrogram, noise_threshold , amp_threshold, normalize_time = False, binarize = False): # threshold =0.45 performs well in west; in east it should be smaller being set as a threshold
    print(f'spectrogram shape: {spectrogram.shape}')  # (721, 264)
    spectrogram = np.array(spectrogram, dtype=float)
    denoised_spec = np.zeros((spectrogram.shape[0], spectrogram.shape[1]), dtype=spectrogram.dtype)
    print(f'denoised_spec shape: {denoised_spec.shape}')  # (721, 528)

    print(f'denoised_spec shape: {denoised_spec.shape}')  # (721, 528)

    # Process each time window
    for t in range(spectrogram.shape[1]):
        # get the current window
        frame = spectrogram[:, t].copy()
        # non_zero values are selected and set the threshold as 25%
        # non_zero values are selected and set the threshold as 25%
        non_zero = frame[frame > 0]
        #print(f'non_zero shape: {non_zero.shape}')
        #print(f'non_zero shape: {non_zero.shape}')
        if len(non_zero) > 0:
            global_threshold = np.percentile(non_zero, noise_threshold * 100) 
            global_threshold = np.percentile(non_zero, noise_threshold * 100) 
            denoised_frame = np.where(frame > global_threshold, frame, 0)
        else:
           denoised_frame = frame
        denoised_spec[:, t] = denoised_frame

    # denoised_spec = np.power(denoised_spec, 1)
    denoised_spec /= np.max(denoised_spec+ 1e-10) 



    # normalize time
    if normalize_time:
        for f in range(denoised_spec.shape[0]):
            row = denoised_spec[f,:]
            max_val = np.max(row)
      
            if max_val>0:
                denoised_spec[f,:] = row/max_val
    # reassign amplitude in the denoised_spec
    if binarize:
        reassignAmp_denoisedSpec = np.where(denoised_spec > amp_threshold, 1, 0)
        return reassignAmp_denoisedSpec
    # reassign amplitude in the denoised_spec
    if binarize:
        reassignAmp_denoisedSpec = np.where(denoised_spec > amp_threshold, 1, 0)
        return reassignAmp_denoisedSpec
    else:
        return denoised_spec
    # return denoised_spec


        return denoised_spec
    # return denoised_spec



if (__name__ == "__main__"):
    # freq_fml = "(cond ((&& (>= t 0.09085753721778567) (<= t 0.15900069013112492)) (cos (cos (neg (tan (sqrt t)))))) ((&& (>= t 0.287715534522988) (<= t 0.3407157645666963)) (* (cos (tan (* (cos 3) (sqrt t)))) (cos 5))) ((&& (>= t 0.43157330178448194) (<= t 0.49971645469782117)) (/ (/ (/ (cos t) (/ t 4)) 4) 7)) ((&& (>= t 0.575431069045976) (<= t 0.5905739919156069)) (/ t (+ 4 (/ (* (exp t) t) 9)))) ((&& (>= t 0.6435742219593151) (<= t 0.6511456833941306)) (tan (neg t))) ((&& (>= t 0.7420032206119163) (<= t 0.8707180650037794)) (tan (* (sin (+ (exp t) (exp (sin 5)))) t))) ((&& (>= t 0.9085753721778567) (<= t 0.9994329093956423)) (sqr (neg (sin (* (tan 10) (sin 7)))))) ((&& (>= t 1.0827189851786125) (<= t 1.1811479838312138)) (sqrt (cos t))) ((&& (>= t 1.2038623681356602) (<= t 1.3250057510927078)) (* (sin (sin (+ (+ (sin t) 2) 6))) t)) ((&& (>= t 1.3552915968319696) (<= t 1.4537205954845707)) (sin 3)) ((&& (>= t 1.4840064412238325) (<= t 1.5597210555719874)) (tan (sqrt (/ (* (- t (sqrt t)) t) t)))) ((&& (>= t 1.6051498241808801) (<= t 1.620292747050511)) (sqr (sin 9))) ((&& (>= t 1.6430071313549575) (<= t 1.7111502842682969)) (+ (* (exp 9) (/ (sqrt (sin 8)) t)) t)) ((&& (>= t 1.7490075914423742) (<= t 1.908008281573499)) (sqr (sqrt (neg (neg (sqr (sin (sin t)))))))) ((&& (>= t 1.9458655887475764) (<= t 2.036723125965362)) (sin t)) ((&& (>= t 2.0745804331394395) (<= t 2.195723816096487)) (sqr (/ (+ (exp (sqrt (sin (sqrt 10)))) t) t))) ((&& (>= t 2.2335811232705645) (<= t 2.294152814749088)) (sqrt (sin (/ (neg (sqr t)) (sin (+ 5 t)))))) ((&& (>= t 2.4152961977061356) (<= t 2.4758678891846597)) (+ (- (- (/ 1 9) 3) (tan t)) t)) (else 0))"
    # mat_fml = sio.loadmat('./plots/formulae.mat')
    # amp_formulae = mat_fml['ampfml_lst']
    fml = '(cond ((&& (>= t 0.2350406062318431) (<= t 0.2653684263907906)) (+ (neg (- (tan t) (- f f))) t)) ((&& (>= t 0.3866797070265806) (<= t 0.424589482225265)) (/ (sin (sin (sqr (sqr (neg (- 5 t)))))) 5)) ((&& (>= t 0.6293022682981606) (<= t 0.6823759535763188)) (tan (tan (neg 1)))) ((&& (>= t 0.6975398636557925) (<= t 0.7202857287750031)) (- (sin (neg t)) (* (+ (exp f) t) t))) ((&& (>= t 0.8036872342121087) (<= t 0.8567609194902669)) (sin (* (* (cos (tan t)) f) t))) ((&& (>= t 0.8946706946889512) (<= t 0.9174165598081618)) (neg (- (* (tan f) t) f))) ((&& (>= t 1.0159819753247412) (<= t 1.0235639303644781)) (exp (/ (* 1 3) (neg t)))) ((&& (>= t 1.0690556606028994) (<= t 1.0766376156426363)) (* (- (sqrt t) (cos 6)) (sqrt t))) ((&& (>= t 1.152457166040005) (<= t 1.1600391210797418)) (tan (sqr (* (sqr (sin (sqrt f))) 1)))) ((&& (>= t 1.1752030311592156) (<= t 1.2586045365963212)) (exp (- (+ 4 (- (cos (sqrt t)) 2)) 8))) ((&& (>= t 1.5618827381857963) (<= t 1.57704664826527)) (* (exp (neg (exp (+ (sqr (exp f)) 6)))) t)) ((&& (>= t 1.5922105583447437) (<= t 1.6680301087421125)) (* (neg (sqr (sqr (exp t)))) (sqrt (tan (sqr 4))))) ((&& (>= t 1.6831940188215861) (<= t 1.7741774792984286)) (cos (cos (- (sqrt (sin 4)) (exp (cos (sin 8))))))) ((&& (>= t 1.8575789847355344) (<= t 1.9713083103315874)) (cos (neg 4))) ((&& (>= t 2.001636130490535) (<= t 2.2215128266429045)) (tan (/ (cos (sqrt (- (sin (sqr t)) f))) f))) ((&& (>= t 2.2821684669607993) (<= t 2.289750422000536)) (sqrt (/ (- t (tan (sqr (sqrt (sqr t))))) 7))) ((&& (>= t 2.312496287119747) (<= t 2.373151927437642)) (* (sqr (neg (* (neg (cos t)) t))) f)) (else 0))'
    # print(amp_fml)
    # print(fml2latex(freq_fml))








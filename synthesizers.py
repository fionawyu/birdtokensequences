## This file takes a .wav file and tries to come up with a formula to mimic the amplitude at each time step
# this is for species, call variants, mix  
# The imports
import numpy as np
from math import isinf, isnan, floor
from grammar import T, U, B, randFML, fMutate, fSize, genUnaryFML
from funcWrappers import wrapExtParseEval, wrapExtParseEvalTwoVar, wrapExtParseEvalBatch, wrapExtParseEvalTwoVarBatch, wrapExtEvalRPN
from utils import lispToRPN
from time import time
import random
import os
from scipy.ndimage import gaussian_filter, maximum_filter1d
# import torch
# import tensorflow as tf
# parameters for bandify and connect_dots_heatmap
bandify_kwargs = dict(bw_bins=10, blur_f=0.2, blur_t=0.6)
connect_kwargs = dict(freq_sigma=0.2, time_sigma=3.0, bridge_bins=3, thresh=0.07)
# Set global seed
def set_seed(seed: int = 47):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # try:
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # except ImportError:
    #     pass

    # try:
    #     tf.random.set_seed(seed)
    # except ImportError:
    #     pass
set_seed()
##################### Public Functions ###################
# def busIncremental(audio_data, fmlLenLim = 3, numIters = 5, verbose = False):
#     # Synthesize first formula
#     fmls = genUnaryFML(average_freq,fmlLenLim)

#     # evaluate the formulae and sort them
#     datalosses = [0] * len(fmls)
#     for fmlI in range(len(fmls)): datalosses[fmlI] = loss(audio_data, fmls[fmlI])
#     fmlIs = [index for index, value in sorted(enumerate(datalosses), key = lambda x: x[1])]
#     fmls = [fmls[fmlI] for fmlI in fmlIs]
#     datalosses = sorted(datalosses)

#     # get the current best formula
#     bestFml = fmls[0]
#     bestLoss = datalosses[0]
#     historicalBestFml = bestFml
#     historicalBestLoss = bestLoss

#     if (verbose): print(f"bestFml: {bestFml}, bestLoss: {bestLoss}")

#     # the main loop
#     for iterI in range(numIters - 1): # stop condition 
#         # evaluate the formulae and sort them
#         datalosses = [0] * len(fmls)
#         for fmlI in range(len(fmls)): datalosses[fmlI] = loss(audio_data, f"(+ {bestFml} {fmls[fmlI]})") # T and U are iterated, B only selects +
#         fmlIs = [index for index, value in sorted(enumerate(datalosses), key = lambda x: x[1])]
#         fmls = [f"(+ {bestFml} {fmls[fmlI]})" for fmlI in fmlIs]
#         datalosses = sorted(datalosses)

#         # the current best fml and dataloss
#         bestFml = fmls[0]
#         bestLoss = datalosses[0]
        
#         # get the current best formula
#         if (datalosses[0] < historicalBestLoss):
#             historicalBestFml = bestFml
#             historicalBestLoss = bestLoss

#         if (verbose): print(f"bestFml: {bestFml}, bestLoss: {bestLoss}")

#     # return the best formula and the dataloss
#     if (verbose): print(f"historicalBestFml: {historicalBestFml}, historicalBestLoss: {historicalBestLoss}")
#     return historicalBestFml, historicalBestLoss


def geneticProgram(times, spec, timelim, thickness, lyFreqlim, uyFreqlim,
                   fmlSeed = "", popSize = 20, numGenLim = 200, fmlSizeLim = 10, verbose = True):
    # Preliminaries
    allFmlsSeen = []
    pop = []
    curLosses = [0] * popSize
    # print(f'shape of spec: {spec.shape}') 

    # # compute average frequency
    # all_h = []
    # for i in range(spec.shape[1]):
    #     for j in range(spec.shape[0]):
    #         if (spec[j,i] > 0):
    #             all_h.append(i)
    # average_freq = sum(all_h) / len(all_h)
 
    # seed the population
    # print(f"ffffffmlSeed: {fmlSeed}") # nothing
    # with the derivative guard on, initial candidates whose score would be
    # negative infinity are redrawn, so the parent is admissible from the
    # first generation on (a seed is subject to the same check)
    sampleTs = _sampleTimes(times, spec)
    def admissible(fml):
        return (not DERIV_GUARD) or fmlDerivsFinite(fml, sampleTs)
    if (len(fmlSeed) > 0 and admissible(fmlSeed)):
        allFmlsSeen.append(fmlSeed)
        pop.append(fmlSeed)

    # Generate initial population
    for i in range(len(pop), popSize):
        while (1):
            fml = randFML(fmlSizeLim)
            if (fml not in allFmlsSeen):
                allFmlsSeen.append(fml)
                if (not admissible(fml)):
                    GUARD_STATS["init_redrawn"] += 1
                    continue
                pop.append(fml)
                break

    # sanity check
    assert(len(pop) == popSize and len(pop) == len(curLosses))

    # init the best formula
    bestFml = pop[0]
    bestLoss = specLoss(bestFml, times, spec, lyFreqlim, uyFreqlim,thickness)
    if (verbose): print(f"bestfml: {bestFml} best dataloss: {bestLoss}")

    # the main loop
    # genI = 0
    startT = time()
    # while (time() - startT < timelim):
        # genI += 1
    for genI in range(numGenLim):
        if (verbose): print(f"time used: {time() - startT}")
        # evaluate the population
        for pI in range(popSize):
            curLosses[pI] = specLoss(pop[pI], times, spec, lyFreqlim, uyFreqlim, thickness)

        # sort the datalosses and the population
        popI = [index for index, value in sorted(enumerate(curLosses), key = lambda x: x[1])]
        pop = [pop[i] for i in popI]
        curLosses = sorted(curLosses)

        # update the best formula
        if (verbose): print(f"genI: {genI} | cur best dataloss: {curLosses[0]}, cur best fml: {pop[0]} | {len(allFmlsSeen)} unique formulae tried")
        if (curLosses[0] < bestLoss):
            bestFml = pop[0]
            bestLoss = curLosses[0]
            if (verbose): print(f"new best: {bestFml}, dataloss: {bestLoss}")

        # generate the next generation
        for pI in range(1, popSize):
            # redraw from the parent on a duplicate, so every child is
            # exactly fMutate's k operations away from bestFml
            while (1):
                newFml = fMutate(bestFml, fmlSizeLim)
                if (newFml not in allFmlsSeen):
                    allFmlsSeen.append(newFml)
                    pop[pI] = newFml
                    break
    # print(f"freq generation in total: {genI} ")
    GUARD_STATS["segments"].append({
        "n_frames": int(spec.shape[1]), "best_loss": float(bestLoss),
        "best_admissible": bool(fmlDerivsFinite(bestFml, sampleTs))})
    # return the best formula and the dataloss
    return bestFml, bestLoss


def geneticProgramAmp(times, spec, freq_spec, timelim, lyFreqlim, uyFreqlim,
                   fmlSeed = "", popSize = 20, numGenLim = 1, fmlSizeLim = 10, verbose = True):
    # Preliminaries
    allFmlsSeen = []
    pop = []
    curLosses = [0] * popSize

    # # compute average frequency
    # all_h = []
    # for i in range(spec.shape[1]):
    #     for j in range(spec.shape[0]):
    #         if (spec[j,i] > 0):
    #             all_h.append(i)
    # average_freq = sum(all_h) / len(all_h)
    # seed the population
    if (len(fmlSeed) > 0): pop.append(fmlSeed)

    # Generate initial population
    for i in range(len(pop), popSize):
        while (1):
            fml = randFML(fmlSizeLim, "amp")
            if (fml not in allFmlsSeen):
                allFmlsSeen.append(fml)
                pop.append(fml)
                break

    # sanity check
    assert(len(pop) == popSize and len(pop) == len(curLosses))

    # init the best formula
    bestFml = allFmlsSeen[0]
    bestLoss = ampspecLoss(bestFml, times, spec, freq_spec, lyFreqlim, uyFreqlim)
    if (verbose): print(f"bestfml: {bestFml} best dataloss: {bestLoss}")

    # the main loop
    # genI = 0
    startT = time()
    # while (time() - startT < timelim):
        # genI += 1
    for genI in range(numGenLim):
        # if (verbose): print(f"generation: {genI}")
        # evaluate the population
        for pI in range(popSize):
            curLosses[pI] = ampspecLoss(pop[pI], times, spec, freq_spec, lyFreqlim, uyFreqlim)

        # sort the datalosses and the population
        popI = [index for index, value in sorted(enumerate(curLosses), key = lambda x: x[1])]
        pop = [pop[i] for i in popI]
        curLosses = sorted(curLosses)

        # update the best formula
        if (verbose): print(f"cur best dataloss: {curLosses[0]}, cur best fml: {pop[0]}")
        if (curLosses[0] < bestLoss):
            bestFml = pop[0]
            bestLoss = curLosses[0]
            if (verbose): print(f"new best: {bestFml}, dataloss: {bestLoss}")

        # generate the next generation
        for pI in range(1, popSize):
            # redraw from the parent on a duplicate, so every child is
            # exactly fMutate's k operations away from bestFml
            while (1):
                newFml = fMutate(bestFml, fmlSizeLim, mode = "amp")
                if (newFml not in allFmlsSeen):
                    allFmlsSeen.append(newFml)
                    pop[pI] = newFml
                    break
    # print(f"amp generation in total: {genI} ")
    # return the best formula and the dataloss
    return bestFml, bestLoss

# Computes the dataloss of a given formula
# def loss(audio_data, fml, regFlag = True):
#     dataloss = 0
#     dataSize = len(audio_data)
#     for t in range(dataSize):
#         amp = wrapExtParseEval(fml, t)
#         # wrapExtParseEval: 
#         if (isinf(amp) or isnan(amp)):
#             dataloss = float("inf")
#             break
#         dataloss += abs(audio_data[t] - amp)
#     avgLoss = dataloss / dataSize
#     if (regFlag): avgLoss += 0.001 * fSize(fml) 
#     return avgLoss
# def loss(audio_data, fml, regFlag = True):
#     individual_losses = []
#     dataSize = len(audio_data)
#     for t in range(dataSize):
#         amp = wrapExtParseEval(fml, t)
#         if (isinf(amp) or isnan(amp)):
#             return float("inf")
#         individual_losses.append(abs(audio_data[t] - amp))
    
#     # Calculate median loss instead of average
#     median_loss = np.median(individual_losses)
    
#     # Add regularization if needed
#     if (regFlag): 
#         median_loss += 0.001 * fSize(fml)
        
#     return median_loss
def loss(audio_data, fml, regFlag = True):
    dataloss = 0
    dataSize = len(audio_data)
    for t in range(dataSize):
        amp = wrapExtParseEval(fml, t)
        # wrapExtParseEval: 
        if (isinf(amp) or isnan(amp)):
            dataloss = float("inf")
            break
        dataloss += abs(audio_data[t] - amp)
    avgLoss = dataloss / dataSize
    if (regFlag): avgLoss += 0.001 * fSize(fml) 
    return avgLoss

def fmlDrawSpec(fml, times, spec, lyFreqlim, uyFreqlim, thickness=10):
    draw_val = 1
    freq_spec = np.zeros(spec.shape)
    height, width = freq_spec.shape
    # width = 10000
    # height = 10000

    #normed_freqs = np.linspace(lyFreqlim/1000, uyFreqlim/10000, height)
    # normed_freqs = np.linspace(0, 1, height)
    normed_freqs = np.linspace(0, uyFreqlim/12000, height)
    #print(lyFreqlim, uyFreqlim)
    # normlize times to [0, 1]
    # normed_times = np.linspace(0, 50, width) 
    # normed_times = np.linspace(0.0, 1.0, width)
    normed_times = np.linspace(times[0], times[-1], width)
    # normalize times to [0, 50]


    # predict the frequencies using the batch function
    #pred_freqs = wrapExtParseEvalBatch(fml, normed_times)
    rpn = lispToRPN(fml)
    if (rpn == None):
        return freq_spec
    pred_freqs, _ = wrapExtEvalRPN(rpn, normed_times)

    for t in range(width):
        pred_freq = pred_freqs[t]
        pred_freq = np.abs(normed_freqs - pred_freq).argmin()

        if (isnan(pred_freq)): pred_freq = 0
        elif (isinf(pred_freq)): pred_freq = height
        else: pred_freq = max(min(pred_freq, height), 0)

        # no draw if predicted frequency is 0
        if (pred_freq == 0): continue

        # Draw the line with thickness
        leftVal = -int(thickness // 2)
        rightVal = int(thickness // 2)

        yl = max(0, floor(pred_freq) + leftVal)
        yu = min(height, floor(pred_freq) + rightVal)
        xl = max(0, t + leftVal)
        xu = min(width, t + rightVal)
        freq_spec[yl:yu, xl:xu] = draw_val

        '''
        for e_x in range(leftVal, rightVal + 1):
            for e_y in range(leftVal, rightVal + 1):
                y, x = floor(pred_freq) + e_y, t + e_x
                if (0 <= x < width and 0 <= y < height):
                    freq_spec[y, x] = draw_val
        '''

    return freq_spec


# Derivative guard (Section 2.4): slope and curvature are the mean first
# derivative and mean |second derivative| of the fitted formula, so a
# candidate is only admissible if f, f' and f'' are all defined, finite and
# real at every sampled time point. Inadmissible candidates get an infinite
# loss (i.e. a fitness score of negative infinity). Set to False to
# reproduce runs made before the guard existed.
DERIV_GUARD = True

# bookkeeping for the guard, filled by geneticProgram/specLoss and read by the
# batch drivers (e.g. run_consistency_recording.py) after each seed
GUARD_STATS = {}
def resetGuardStats():
    GUARD_STATS.clear()
    GUARD_STATS.update(evaluated = 0, rejected = 0, init_redrawn = 0, segments = [])
resetGuardStats()

def _parseFml(fml):
    toks = fml.replace("(", " ( ").replace(")", " ) ").split()
    def rec(pos):
        if toks[pos] != "(": return toks[pos], pos + 1
        node, pos = [], pos + 1
        while toks[pos] != ")":
            child, pos = rec(pos)
            node.append(child)
        return node, pos + 1
    node, pos = rec(0)
    if pos != len(toks): raise ValueError(f"trailing tokens in {fml}")
    return node

# second-order forward-mode AD over a numpy array of time points:
# returns (f, f', f'') as arrays. Points where an operator is undefined or
# not differentiable come out as nan/inf: sqrt/log of a non-positive argument
# (sqrt(0) is defined but its derivative is not), abs at its kink, division
# by zero, and exp/tan overflow.
def _evalJet(node, ts):
    if isinstance(node, str):
        if node == "t": return ts, np.ones_like(ts), np.zeros_like(ts)
        c = float(node)
        return np.full_like(ts, c), np.zeros_like(ts), np.zeros_like(ts)
    head, args = node[0], node[1:]
    if (head in U):
        v, d, dd = _evalJet(args[0], ts)
        if (head == "sqr"): return v * v, 2 * v * d, 2 * d * d + 2 * v * dd
        if (head == "neg"): return -v, -d, -dd
        if (head == "exp"):
            e = np.exp(v)
            return e, e * d, e * (d * d + dd)
        if (head == "sin"):
            s, c = np.sin(v), np.cos(v)
            return s, c * d, -s * d * d + c * dd
        if (head == "cos"):
            s, c = np.sin(v), np.cos(v)
            return c, -s * d, -c * d * d - s * dd
        if (head == "tan"):
            tv = np.tan(v)
            s2 = 1 + tv * tv
            return tv, s2 * d, 2 * tv * s2 * d * d + s2 * dd
        if (head == "sqrt"):
            v = np.where(v > 0, v, np.nan)
            r = np.sqrt(v)
            return r, d / (2 * r), -d * d / (4 * v * r) + dd / (2 * r)
        if (head == "log"):
            v = np.where(v > 0, v, np.nan)
            return np.log(v), d / v, -d * d / (v * v) + dd / v
        if (head == "abs"):
            sgn = np.where(v != 0, np.sign(v), np.nan)
            return np.abs(v), sgn * d, sgn * dd
    elif (head in B):
        (av, ad, add_), (bv, bd, bdd) = _evalJet(args[0], ts), _evalJet(args[1], ts)
        if (head == "+"): return av + bv, ad + bd, add_ + bdd
        if (head == "-"): return av - bv, ad - bd, add_ - bdd
        if (head == "*"): return av * bv, ad * bv + av * bd, add_ * bv + 2 * ad * bd + av * bdd
        if (head == "/"):
            bv = np.where(bv != 0, bv, np.nan)
            v = av / bv
            d = (ad - v * bd) / bv
            return v, d, (add_ - 2 * d * bd - v * bdd) / bv
    raise ValueError(f"no derivative rule for operator {head}")

# True if fml, its first and its second derivative are all defined, finite
# and real at every point of ts
def fmlDerivsFinite(fml, ts):
    try:
        node = _parseFml(fml)
    except (IndexError, ValueError):
        return False
    with np.errstate(all = "ignore"):
        jet = _evalJet(node, np.asarray(ts, dtype = float))
    return all(np.all(np.isfinite(x)) for x in jet)


# the time points fmlDrawSpec samples a formula at
def _sampleTimes(times, spec):
    return np.linspace(times[0], times[-1], spec.shape[1])


def specLoss(fml, times, spec, lyFreqlim, uyFreqlim, thickness):
    GUARD_STATS["evaluated"] += 1
    if (DERIV_GUARD and not fmlDerivsFinite(fml, _sampleTimes(times, spec))):
        GUARD_STATS["rejected"] += 1
        return float("inf")
    pred_spec = fmlDrawSpec(fml, times, spec, lyFreqlim, uyFreqlim, thickness)
    masked_spec = spec.copy()
    masked_spec[spec < 0.15] = 0
    masked_spec[spec >= 0.15] = 1

    diff_mat = masked_spec - pred_spec
    diff_mat[(masked_spec == 0) & (pred_spec != 0)] = 0.15

    real_loss = np.mean(np.abs(diff_mat))
    #return real_loss + 0.001 * real_loss * fSize(fml)
    return real_loss + 0.001 * np.log(fSize(fml))
    #return real_loss


def ampflmDraw(fml, times, spec, freq_spec, lyFreqlim, uyFreqlim, bandify_kwargs=None, connect_kwargs=None, mask_thresh=0.0):
    H, W = spec.shape
    bandify_kwargs = bandify_kwargs or {}
    connect_kwargs = connect_kwargs or {}

    # normalized axes for 2-var evaluation (broadcast shapes (H,1) & (1,W))
    normed_freqs = np.linspace(lyFreqlim/1000, uyFreqlim/1000, H, endpoint=False)[:, None]
    # normed_times = np.linspace(0.0, 1.0, W, endpoint=False)[None, :]
    normed_times = np.linspace(times[0], times[-1], W, endpoint=False)[None, :]
    # print(f'ampflmDraw normed_times: {normed_times}')
    # print(freq_spec.shape)
    # spec_min = np.min(spec)
    # spec_max = np.max(spec)
    # # Normalize spec to [0, 1] first
    # spec_norm = (spec - spec_min) / (spec_max - spec_min)

    # Scale to [0.1, 1]
    # spec_norm = spec_norm * 0.9 + 0.1

    # compute all amplitudes as a whole
    pred_amps = wrapExtParseEvalTwoVarBatch(fml, normed_freqs, normed_times, freq_spec)

    # for f in range(height):
    #     for t in range(width):
    #         # Skip if there's no energy at this bin/time
    #         if freq_spec[f, t] == 0: continue
    #         # get the current amplitude
    #         pred_amp = pred_amps[f, t]
    #         # Check for non-numerical results
    #         if np.isnan(pred_amp):
    #             pred_amp = 0.1
    #         elif np.isinf(pred_amp):
    #             pred_amp = 1.0
    #         else:
    #             # Ensure pred_amp is within [0.1, 1] before further normalization
    #             pred_amp = max(min(pred_amp, 1.0), 0.1)
    #         # Post-normalization:
    #         # Map the predicted amplitude onto the same scale as spec_norm.
    #         # This assumes that pred_amp is in a similar range as the original spec.
    #         pred_amp_norm = (pred_amp - 0.1) / (1.0 - 0.1)  # equivalent to (pred_amp - 0.1) / 0.9
    #         amppredicted_spec_norm[f, t] = pred_amp_norm
     # clean + clamp to [0.1, 1.0]
    pred_amps = np.nan_to_num(pred_amps, nan=0.1, posinf=1.0, neginf=0.1)
    pred_amps = np.clip(pred_amps, 0.1, 1.0)

    # map to [0,1]
    ampnorm = (pred_amps - 0.1) / 0.9

    # apply mask (keep only bins with energy)
    mask = (freq_spec > mask_thresh)
    # ampnorm = np.where(mask, ampnorm, 0.0)
    amppredicted_spec = np.where(mask, ampnorm, 0.0)

    # # post-process ONCE for the whole map
    # amppredicted_spec = _bandify(ampnorm, **bandify_kwargs)
    # amppredicted_spec = _connect_dots_heatmap(ampnorm, **connect_kwargs)
            
    return amppredicted_spec


def ampspecLoss(fml, times, spec, freq_spec, lyFreqlim, uyFreqlim):
    amppredicted_spec = ampflmDraw(fml, times, spec, freq_spec,lyFreqlim, uyFreqlim)
    return np.sum(np.abs(spec - amppredicted_spec)) + 0.01 * fSize(fml)


def totalLoss(specfml, ampfml, times, spec, freq_spec, thickness):
    return ampspecLoss(specfml, times, spec, freq_spec,lyFreqlim, uyFreqlim) + specLoss(ampfml, times, spec, lyFreqlim, uyFreqlim, thickness)

def _bandify(S_pred, bw_bins=8, blur_f=1.2, blur_t=0.6):
    """
    S_pred: (freq x time) linear-magnitude spectrogram (numpy array)
    Returns: softened band-like spectrogram (0..1)
    """
    S = np.maximum(S_pred, 0).astype(float)
    if S.max() > 0:
        S /= S.max()

    F, T = S.shape

    # center of the ridge per frame
    f0 = S.argmax(axis=0)

    # Gaussian band around the center (bright middle, attenuated edges)
    f = np.arange(F)[:, None]
    sigma = bw_bins / np.sqrt(8*np.log(2))  # FWHM≈bw_bins -> sigma
    G = np.exp(-0.5 * ((f - f0[None, :]) / sigma)**2)

    # mix original with band mask (keep edges but lighter)
    S = S * (0.3 + 0.7 * G)

    # light smoothing so edges aren’t harsh
    S = gaussian_filter(S, sigma=(blur_f, blur_t), mode='nearest')

    # normalize back to 0..1
    S /= (S.max() + 1e-12)
    return S



def _connect_dots_heatmap(S_pred, freq_sigma=0.15, time_sigma=2.0,
                         bridge_bins=2, thresh=0.08):
    """
    S_pred: (F x T) linear magnitude (0..1 recommended)
    1) anisotropic Gaussian blur (time >> freq)
    2) max-filter along time to bridge short gaps
    3) simple threshold to keep only connected ridges
    """
    S = np.maximum(S_pred, 0).astype(float)
    if S.max() > 0: S /= S.max()

    # 1) smooth (more along time)
    S = gaussian_filter(S, sigma=(freq_sigma, time_sigma), mode='nearest')

    # 2) connect short temporal holes
    S = maximum_filter1d(S, size=2*bridge_bins+1, axis=1, mode='nearest')

    # 3) keep salient structures
    S_conn = np.where(S >= thresh, S, 0.0)
    if S_conn.max() > 0: S_conn /= S_conn.max()
    return S_conn


if __name__ == '__main__':
    freq_fml = "(cond ((&& (>= t 0.0) (<= t 0.2799054373522459)) (tan (sin (cos (exp (sqrt (tan (neg (neg (cos t)))))))))) ((&& (>= t 0.4085106382978724) (<= t 0.6884160756501182)) (/ (tan 10) (sqrt (+ (neg 6) (sqr 9))))) ((&& (>= t 0.8094562647754138) (<= t 1.0893617021276596)) (neg (sqrt (exp (tan (+ (neg t) (cos (sin 10)))))))) ((&& (>= t 1.210401891252955) (<= t 1.497872340425532)) (/ (tan (- (sqr (cos (tan (neg 5)))) t)) V)) ((&& (>= t 1.6189125295508275) (<= t 1.8988179669030734)) (sqrt (/ (/ (sqrt V) V) t))) ((&& (>= t 2.019858156028369) (<= t 2.307328605200946)) (sqrt (cos (tan (exp (/ t V)))))) ((&& (>= t 2.428368794326241) (<= t 2.7082742316784874)) (neg (+ (tan (/ (tan 9) (+ t t))) V))) ((&& (>= t 2.8293144208037826) (<= t 3.1167848699763594)) (+ (cos (neg 4)) (- (exp (neg (sin t))) V))) (else 0))"
    amp_fml = "(cond ((&& (>= t 0.0) (<= t 0.2799054373522459)) (tan V)) ((&& (>= t 0.4085106382978724) (<= t 0.6884160756501182)) (sqr (sin (/ 2 f)))) ((&& (>= t 0.8094562647754138) (<= t 1.0893617021276596)) (neg (cos (sqrt V)))) ((&& (>= t 1.210401891252955) (<= t 1.497872340425532)) (neg (+ (+ t (tan (neg V))) t))) ((&& (>= t 1.6189125295508275) (<= t 1.8988179669030734)) (- (sin (sqr (sqrt (* t (sin (exp V)))))) t)) ((&& (>= t 2.019858156028369) (<= t 2.307328605200946)) (sqrt (sqr (sin (cos (tan (- (tan f) t))))))) ((&& (>= t 2.428368794326241) (<= t 2.7082742316784874)) (sqrt (- (neg V) (sqr f)))) ((&& (>= t 2.8293144208037826) (<= t 3.1167848699763594)) f) (else 0))"


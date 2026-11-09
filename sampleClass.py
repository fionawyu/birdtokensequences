
# Classify token trends (ZERO/FLAT/UP/DOWN/WIGGLE) per sub-formula over real-time segments.
from doctest import debug
import numpy as np
from funcWrappers import (
    wrapExtGetHeadAndInputs,
    wrapExtEvalRPN,
)
from utils import lispToRPN
# Configurable thresholds
FLAT_RANGE  = 0.0001 # amplitude range threshold for FLAT
# SLOPE_THR   = 0.90   # end-start slope threshold for UP/DOWN
MONO_THR    = 0.90   # fraction of monotone steps for UP/DOWN
SLOPE_FRAC_THR = 0.80
ZERO_LITS   = {"0", "0.0", "0.00"}
ZERO_BOUND = 0.0001
# One evaluation point per spectrogram frame of the segment. The drivers build
# spectrograms with utils.create_audio_spectrogram: 44.1 kHz, 5 ms window,
# hop = window // 4, so one frame lasts FRAME_SEC seconds.
FRAME_SEC = (int(44100 * 5 / 1000) // 4) / 44100

def num_points(W):
    """Number of spectrogram frames in a segment of duration W seconds."""
    return max(2, int(round(W / FRAME_SEC)))

# Segment length computation
def seg_W_fixed(start_sec, end_sec):
    seg_dur = max(0.0, float(end_sec) - float(start_sec))
    actual_seg = (0.0, seg_dur)
    # print(f"Segment: {start_sec}-{end_sec}, W: {seg_dur}, Actual Segment: {actual_seg}")
    return seg_dur, actual_seg  # small safety floor

# Evaluate on real time in [start_sec, end_sec)
def _evalY_on_segment(subfml, W, start_sec=None, end_sec=None):
    """
    Evaluate subfml at W points on real time [start_sec, end_sec), end exclusive.
    Falls back to normalized [0,1) if bounds are not provided.
    """
    t = np.linspace(0, W, num_points(W), endpoint=False)
    # print(f"Evaluating subfml: {subfml} on t values: {t}")
    rpn = lispToRPN(subfml)
    y_raw, _ = wrapExtEvalRPN(rpn, t)  # vectorized over t
    y = np.asarray(y_raw, dtype=float).reshape(-1)
    # print(f"Raw evaluated y values: {y}")
    y = np.nan_to_num(y, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return y

# Zero and constant checks
def hasT_sub(subfml):
    """
    True if expression structurally depends on 't'.
    """
    head, inputs = wrapExtGetHeadAndInputs(subfml)
    s = str(subfml).strip()
    if s == "t" or head == "t":
        return True
    for x in inputs:
        sx = str(x).strip()
        if sx == "t" or hasT_sub(x):
            return True
    return False

# Constant value evaluation
def const_value_sub(subfml, W, atol=1e-9):
    """
    If expression has no 't' and is numerically constant, return its value; else None.
    """
    if hasT_sub(subfml):
        return None
    t = np.linspace(0, W, num_points(W), endpoint=False)
    rpn = lispToRPN(subfml)
    y_raw, _ = wrapExtEvalRPN(rpn, t)  # vectorized over t
    y = np.asarray(y_raw, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=np.inf, posinf=np.inf, neginf=np.inf)
    if not np.isfinite(y).all():
        return None
    if np.ptp(y) <= atol:
        return float(np.mean(y))
    return None

# Zero checks
def isZero_sub(subfml, W, ZERO_BOUND=1e-8):
    s = str(subfml).strip()
    # print(s)
    if s in ZERO_LITS:
        return True
     # Structural safe rules
    head, inputs = wrapExtGetHeadAndInputs(subfml)
    if head == "+" and len(inputs) >= 1:
        return all(isZero_sub(x, ZERO_BOUND) for x in inputs)
    if head == "-" and len(inputs) == 2:
        return isZero_sub(inputs[0], ZERO_BOUND) and isZero_sub(inputs[1], ZERO_BOUND)
    if head == "*" and len(inputs) >= 1:
        return any(isZero_sub(x, ZERO_BOUND) for x in inputs)
    if head == "/" and len(inputs) == 2:
        a, b = inputs
        if isZero_sub(a, ZERO_BOUND):
            vb = const_value_sub(b)
            if vb is not None and abs(vb) > ZERO_BOUND:
                return True
        return False
    if head in {"sqr", "sqrt", "exp","neg", "sin", "tan", "cos", "log", "abs"} and len(inputs) == 1:
        return isZero_sub(inputs[0], ZERO_BOUND)
    # Numeric constant path (covers constant expressions that evaluate to ~0)
    v = const_value_sub(subfml, W, atol=1e-10)
    if v is not None and abs(v) <= ZERO_BOUND:
        return True
    return False

def isConstNoT_sub(subfml):
    return not hasT_sub(subfml)

# ---------- Trend decision for 't'-dependent expr ----------
def decideTrend_sub(subfml, W, start_sec=None, end_sec=None,
                     mono_thr=MONO_THR, slope_frac_thr=SLOPE_FRAC_THR, flat_range=FLAT_RANGE):
    # mono_thr/slope_frac_thr/flat_range default to the module constants above
    # (unchanged behavior for every existing caller); overridable for the
    # threshold sensitivity analysis in mapping_rule_sensitivity/.
    y = _evalY_on_segment(subfml, W, start_sec, end_sec)
    rng = float(np.max(y) - np.min(y))
    slope_frac = abs(y[-1]-y[0]) / (max(y)-min(y)) + 1e-12

    if max(y) < ZERO_BOUND and min(y) > -ZERO_BOUND:
        # print(f"Subfml: {subfml} evaluated to near zero range: {rng}")
        # print(f"y values: {y}")
        return "ZERO"

    if rng < flat_range:
        # print(f"Subfml: {subfml} evaluated to flat range: {rng}")
        return "FLAT"

    dy = np.diff(y)
    if dy.size == 0:
        # print(f"Subfml: {subfml} has no variation in y values: {y}")
        return "FLAT"

    s = np.sign(dy)
    s = s[s != 0]
    flips = int(np.sum(s[1:] * s[:-1] < 0))
    if flips >= 2 and rng >= 0.05:
        return "WIGGLE"
    up_frac   = float(np.mean(dy > 0))
    down_frac = float(np.mean(dy < 0))
    slope     = float(y[-1] - y[0])
    slope_frac = abs(slope) / (rng + 1e-12)
    if slope > 0 and up_frac   >= mono_thr and slope_frac >= slope_frac_thr:
        return "UP"
    if slope < 0 and down_frac >= mono_thr and slope_frac >= slope_frac_thr:
        return "DOWN"

    if rng < 0.05:
        return "FLAT"
    return "WIGGLE"



# ---------- Main single-subfml classifier ----------
def classify_subfml(subfml, W, start_sec=None, end_sec=None,
                     mono_thr=MONO_THR, slope_frac_thr=SLOPE_FRAC_THR, flat_range=FLAT_RANGE):
    # No 't' -> constant categories
    if isConstNoT_sub(subfml):
        # print(f"Subfml: {subfml} is constant no t.")
        if isZero_sub(subfml, W):
            # print(f"Subfml: {subfml} is zero.")
            return "ZERO"
        # print(f"Subfml: {subfml} is flat.")
        return "FLAT"
    # Has 't' -> evaluate over the *real time* window
    return decideTrend_sub(subfml, W, start_sec, end_sec, mono_thr, slope_frac_thr, flat_range)


# ---------- Explained variants (figure/table annotation only) ----------
# Mirror classify_subfml/decideTrend_sub exactly but also return which
# rule/threshold comparison fired, as a human-readable string. Kept separate
# from the production functions above so nothing on the classification path
# used by scoring/clustering changes.
def decideTrend_sub_with_rule(subfml, W, start_sec=None, end_sec=None):
    y = _evalY_on_segment(subfml, W, start_sec, end_sec)
    rng = float(np.max(y) - np.min(y))

    if max(y) < ZERO_BOUND and min(y) > -ZERO_BOUND:
        return "ZERO", f"max(y)<{ZERO_BOUND} and min(y)>-{ZERO_BOUND}"

    if rng < FLAT_RANGE:
        return "FLAT", f"range={rng:.2e} < FLAT_RANGE={FLAT_RANGE}"

    dy = np.diff(y)
    if dy.size == 0:
        return "FLAT", "no variation in y (single point)"

    s = np.sign(dy)
    s = s[s != 0]
    flips = int(np.sum(s[1:] * s[:-1] < 0))
    if flips >= 2 and rng >= 0.05:
        return "WIGGLE", f"sign flips={flips}>=2 and range={rng:.3f}>=0.05"

    up_frac = float(np.mean(dy > 0))
    down_frac = float(np.mean(dy < 0))
    slope = float(y[-1] - y[0])
    slope_frac = abs(slope) / (rng + 1e-12)
    if slope > 0 and up_frac >= MONO_THR and slope_frac >= SLOPE_FRAC_THR:
        return "UP", (f"slope>0, up_frac={up_frac:.2f}>=MONO_THR={MONO_THR}, "
                       f"slope_frac={slope_frac:.2f}>=SLOPE_FRAC_THR={SLOPE_FRAC_THR}")
    if slope < 0 and down_frac >= MONO_THR and slope_frac >= SLOPE_FRAC_THR:
        return "DOWN", (f"slope<0, down_frac={down_frac:.2f}>=MONO_THR={MONO_THR}, "
                         f"slope_frac={slope_frac:.2f}>=SLOPE_FRAC_THR={SLOPE_FRAC_THR}")

    if rng < 0.05:
        return "FLAT", f"fallback: range={rng:.3f}<0.05"
    return "WIGGLE", "fallback: none of ZERO/FLAT/UP/DOWN criteria met"


def classify_subfml_with_rule(subfml, W, start_sec=None, end_sec=None):
    """Same decision as classify_subfml, plus a short string naming the
    rule/threshold comparison that produced it."""
    if isConstNoT_sub(subfml):
        if isZero_sub(subfml, W):
            return "ZERO", "constant, no t dependence, value ~= 0"
        return "FLAT", "constant, no t dependence, nonzero value"
    return decideTrend_sub_with_rule(subfml, W, start_sec, end_sec)


def classify_subfmls_by_segments(subfml_list, segments,
                                  mono_thr=MONO_THR, slope_frac_thr=SLOPE_FRAC_THR, flat_range=FLAT_RANGE):
    labels = []
    if len(subfml_list) != len(segments):
        print(f"[warn] subfml_list ({len(subfml_list)}) != segments ({len(segments)}); extra items will be ignored.")

    for (start_sec, end_sec), sub in zip(segments, subfml_list):
        seg_len, actual_seg = seg_W_fixed(start_sec, end_sec)
        lab = classify_subfml(sub, seg_len, start_sec=start_sec, end_sec=end_sec,
                               mono_thr=mono_thr, slope_frac_thr=slope_frac_thr, flat_range=flat_range)
        # print(f"Segment: {start_sec}-{end_sec}, W: {seg_len}, Subfml: {sub}, Label: {lab}")
        labels.append(lab)
    return labels


if __name__ == "__main__":

    #########################YWAR##########################
    # segments = [
    #     (1.251, 1.451),
    #     (3.09, 3.566),
    #     (3.666, 3.991),
    #     (5.292, 6.193),
    #     (6.894, 7.945),
    #     (8.633, 9.734),
    #     (10.447, 10.622),
    #     (10.747, 11.51),
    #     (12.323, 12.961),
    #     (13.049, 13.199),
    #     ]
    # subfml_list =     [
    #     '(sin (+ (sin (+ (* 0.8 t) 1)) 6))',
    #     '(sin (+ (* 0.8 t) 2))',
    #     '(sin (exp (sin (tan (sin (+ (* 0.5 t) 1))))))',
    #     '(sin (+ (* 0.6 t) 2))',
    #     '(abs (sin (+ (sin (+ (* 0.6 t) 3)) 6)))',
    #     '(sin (+ (sin (+ (* 0.9 t) 8)) 2))',
    #     '(sin (+ (sin (+ (* 0.2 t) 3)) t))',
    #     '(sin (+ (sin (+ (* 0.9 (sin t)) 2)) 2))',
    #     '(sin (exp (sin (+ (* 0.9 t) 4))))',
    #     '(sin (+ (sin (+ (* 0.9 t) 4)) 1))',
    #     ]
    ###################chickadee#####################
    # segments = [
    #     (0.813, 1.550),
    #     (2.013, 2.626),
    #     (3.514, 4.726),
    #     (5.589, 7.277),
    #     (8.065, 9.690),
    #     (10.378, 12.141),
    #     (12.791, 14.754),
    # ]

    # subfml_list = [
    #     '(sin (sin (+ (* 2 (sin t)) 1)))',
    #     '(sin (sin (cos (sin (sin (+ (* 4 t) 2))))))',
    #     '0',
    #     '0',
    #     '(neg (sin (sin 6)))',
    #     '(sin (sin (sin (sin (neg 6)))))',
    #     '(sin (sin (sqr (sin 10))))',
    # ]
    ###########feebee#####################
    # segments = [
    #     (1.103, 4.485),
    #     (5.663, 9.309),
    # ]
    # subfml_list = [
    #     '(tan 3.5)',
    #     '(log 1.4)',
    # ]
    ####################TEWAR#####################
    # segments = [
    #     (1.674, 2.574),
    #     (4.073, 4.698),
    #     (6.897, 7.646),
    #     (9.020, 9.520),
    #     (11.269, 12.294),
    #     (13.743, 14.268),
    #     (15.867, 17.341),
    #     (17.916, 19.390),
    #     (20.490, 21.789),
    #     (22.614, 24.163),
    #     (24.938, 25.212),
    #     (25.487, 26.212),
    #     (27.036, 28.386),
    #     (29.410, 29.685),
    #     (29.960, 30.710),
    #     (31.484, 33.158),
    #     (33.983, 35.407),
    #     (36.757, 38.231),
    #     (39.355, 40.930),
    #     (42.129, 43.553),
    #     (44.878, 46.452),
    # ]

    # subfml_list = [
    #     '(sin (exp (* (sin (+ (* 0.9 t) 5)) t)))',
    #     '(sin (+ (* 0.4 t) 9))',
    #     '(sin (sin (exp (sin (+ (* 0.9 t) 2)))))',
    #     '(sin (+ (sin (+ t 7)) 2))',
    #     '(sin (sin (exp (sin (+ (* 0.9 t) 8)))))',
    #     '(sin (+ (* t t) 9))',
    #     '(exp (sin (sin (+ (* 0.8 t) 9))))',
    #     '(sin (sin (+ (sin (+ (* 0.5 t) 0)) 2)))',
    #     '(tan (sin (+ (* 0.4 t) 2)))',
    #     '(sin (sin (+ (sin t) 8)))',
    #     '(exp (sin (+ (* 0.3 t) 0)))',
    #     '(sin (+ t 8))',
    #     '(sin (sin (+ (* 0.5 t) 2)))',
    #     '(exp (sin (+ (* 0.3 t) 0)))',
    #     '(sin (+ (sin (sin (+ t 0))) 8))',
    #     '(sin (sin (exp (sin (+ t 9)))))',
    #     '(exp (* (sin (+ (* 0.1 t) 5)) t))',
    #     '(sin (+ (sin t) 2))',
    #     '(exp (* (sin (+ (* 0.6 t) 4)) t))',
    #     '(exp (* (sin (+ (* 0.7 t) 4)) t))',
    #     '(exp (* (sin (+ (* 0.0 t) 5)) t))',
    # ]
    ####################Tseet#####################
    # segments = [
    #     (0.536, 0.931),
    # ]
    # subfml_list = [
    #     '(sin (+ (* (sin (exp (* 3.0 t))) t) 7))',
    # ]
    #################WTSP###################
    # segments = [
    #     (5.943, 8.203),
    #     (16.006, 19.714),
    #     (20.937, 23.459),
    #     (24.158, 24.758),
    #     (25.419, 27.829),
    #     (28.528, 29.127),
    #     (29.802, 32.324),
    #     (33.060, 33.684),
    #     (34.359, 36.656),
    # ]

    # subfml_list = [
    #     '(sin 2.6)',
    #     '(cos 5.0)',
    #     '(/ (log 2.1) 2.5)',
    #     '(cos 5.0)',
    #     '(cos 5.0)',
    #     '(/ 2.0 6.9)',
    #     '(/ 2.1 7.2)',
    #     '(cos 5.0)',
    #     '(cos 5.0)',
    # ]
    ####################YRWA####################
    segments = [
        (7.102, 7.852),
        (9.202, 9.402),
        (9.527, 9.952),
        (10.328, 11.053),
        (11.653, 11.903),
        (12.578, 12.828),
        (12.928, 13.353),
        (13.803, 14.929),
        (15.004, 15.429),
        (16.029, 16.279),
        (16.404, 16.804),
        (17.329, 18.805),
        (19.430, 19.680),
        (19.805, 20.530),
        (20.730, 22.180),
        (22.881, 23.156),
        (23.231, 23.731),
        (24.231, 24.956),
        (25.056, 25.206),
        (25.406, 25.656),
        (26.332, 26.607),
        (26.782, 27.157),
        (27.482, 28.157),
        (28.832, 28.982),
        (29.882, 30.207),
        (30.458, 30.708),
        (30.983, 31.533),
        (34.759, 34.934),
    ]

    subfml_list = [
        '(sin (+ (sin (sin (+ (* 0.3 t) 3))) 9))',
        '(sin (sin (+ (sin (+ (* 1.5 t) 2)) 8)))',
        '(sin (+ (sin (+ (* 0.8 t) 2)) 6))',
        '(sin (sin (+ (sin (+ (* 0.7 t) 2)) 2)))',
        '(sin (+ (sin (+ (* 1.0 t) 1)) 2))',
        '(exp (sin (+ (* 1.7 t) 10)))',
        '(sin (sin (sin (sin (sin (+ (* 1.5 t) 2))))))',
        '(sin (sin (+ (sin (+ t 4)) 10)))',
        '(sin (+ (* 0.6 t) 9))',
        '(exp (sin (+ (* 1.9 t) 10)))',
        '(sin (+ (sin (+ (* 0.9 t) 2)) 6))',
        '(sin (+ (* 0.1 t) 9))',
        '(exp (sin (+ (* (exp (* 4.0 t)) t) 10)))',
        '(sin (sin (sin (sin (sin (+ (* 2.2 t) 2))))))',
        '(sin (+ (sin (sin (+ (* 2.7 t) 7))) 2))',
        '(sin (sin (+ (sin (+ t 6)) 9)))',
        '(sin (+ (sin (+ (* 0.5 t) 6)) 9))',
        '(sin (sin (exp (sin (+ (* 2.3 t) 4)))))',
        '(sin (+ (sin (+ (* 1.1 t) 7)) 6))',
        '(exp (* 2.0 (sin (+ (* 1.8 t) 10))))',
        '(sin (sin (sin (cos (sin (+ (* 2.2 t) 8))))))',
        '(exp (* 2.0 (sin (+ t 10))))',
        '(sin (sin (+ (sin (+ (* 0.8 t) 2)) 6)))',
        '(sin (+ (sin (+ (* 2.7 t) 8)) 2))',
        '(sin (+ (sin (+ (* 2.1 t) 8)) 8))',
        '(sin (abs (sin (+ (* 0.3 t) 6))))',
        '(sin (exp (sin (+ (* 1.8 t) 10))))',
        '(sin (sin (+ (sin (+ (* 0.9 t) 10)) 1)))',
    ]

    # test in real case 2
    # segments=[(2.539, 2.777), (2.864, 3.002), (4.591, 5.304), (5.942, 7.067), (7.618, 8.769), (9.219, 10.395), (10.532, 10.657), (10.933, 12.146)]
    # freq_subfmls=['(sin (+ (* 0.5 (exp (* 2.0 t))) 8))', '(sin (sqr (sin (+ (* 0.7 t) 4))))', '(sin (+ t 2))', '(exp (* 7.0 (log (sin (+ (* 0.5 t) 7)))))', '(sin (+ (sin (+ (* 0.7 t) 9)) 9))', '(sin (sin (+ (sin (+ (* 0.8 t) 4)) 10)))', '(sin (sin (sin (+ (* 0.5 t) 9))))', '(sin (+ (sin (+ (* 0.8 t) 2)) 6))']
    labels = classify_subfmls_by_segments(subfml_list, segments)
    for subfml, label in zip(subfml_list, labels):
        print(f"Subfml: {subfml}, Label: {label}")







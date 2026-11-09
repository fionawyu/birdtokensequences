import copy

import numpy as np
import pytest

from tokenseq import classify, formula as F, metrics, segmentation, snr, symbolic_regression as SR, tokens
from tokenseq.config import load_config, parse_time_points
from tokenseq.distances import levenshtein

CFG = load_config()


def tok_cfg(**kw):
    c = copy.deepcopy(CFG["tokens"])
    c.update(kw)
    c["time_points"] = parse_time_points(c["time_points"])
    return c


# ---------------------------------------------------------------- formulas

def test_parse_roundtrip_and_size():
    s = "(+ (sin t) 2)"
    node = F.parse(s)
    assert F.to_string(node) == s
    assert F.size(node) == 4                       # +, sin, t, 2 (Sec. 2.4 example)


def test_random_formulas_respect_limits():
    rng = np.random.default_rng(0)
    cfg = CFG["symbolic_regression"]
    for _ in range(300):
        f = SR.random_formula(rng, cfg["max_formula_nodes"], cfg["const_range"])
        assert F.size(f) <= cfg["max_formula_nodes"]
        m = SR.mutate(f, rng, cfg)
        assert F.size(m) <= cfg["max_formula_nodes"]


def test_jet_matches_sympy():
    rng = np.random.default_rng(1)
    t = np.linspace(0.1, 2.0, 25)
    checked = 0
    while checked < 40:
        node = SR.random_formula(rng, 8, (0.0, 10.0))
        if not F.derivatives_finite(node, t):
            continue
        a = F.jet(node, t)
        b = F.sympy_derivatives(node, t)
        if not all(np.all(np.isfinite(x)) for x in b):
            continue
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
        checked += 1


def test_derivative_guard():
    t = np.linspace(0, 1, 10)
    assert not F.derivatives_finite(F.parse("(log (neg t))"), t)
    assert not F.derivatives_finite(F.parse("(sqrt t)"), t)    # f' undefined at t = 0
    assert F.derivatives_finite(F.parse("(sin t)"), t)


def test_ga_fits_a_line():
    cfg = dict(CFG["symbolic_regression"], generations=60)
    t = np.linspace(0, 1, 30)
    res = SR.fit(t, 0.3 + 0.2 * t, cfg, np.random.default_rng(0))
    pred = F.evaluate(F.parse(res.formula), t)
    assert np.mean(np.abs(pred - (0.3 + 0.2 * t))) < 0.1


# ---------------------------------------------------------------- tokens

@pytest.mark.parametrize("fml,expected", [
    ("0.265", tokens.FLAT),
    ("0", tokens.SILENCE),
    ("(* 0.1 t)", tokens.UP),
    ("(- 1 (* 0.1 t))", tokens.DOWN),
    ("(sin (* 30 t))", tokens.WIGGLE),
])
def test_token_rules(fml, expected):
    for mode in ("frames", 8):
        assert tokens.formula_token(fml, 40, 0.0015, 10.0, tok_cfg(time_points=mode)) == expected


def test_time_point_modes():
    assert len(tokens.time_points(57, 0.0015, 10.0, "frames")) == 57
    assert len(tokens.time_points(57, 0.0015, 10.0, 8)) == 8
    assert parse_time_points("8") == 8
    assert parse_time_points("frames") == "frames"
    with pytest.raises(ValueError):
        parse_time_points("1")


# ---------------------------------------------------------------- segmentation

def test_segmentation_merges_and_drops():
    frame_s = 0.001
    spec = np.zeros((20, 400))
    spec[5, 10:60] = 1.0      # 50 ms
    spec[5, 70:120] = 1.0     # gap of 10 ms -> merged with the first
    spec[5, 200:205] = 1.0    # 5 ms -> dropped
    spec[5, 300:350] = 1.0    # separate segment
    ivs = segmentation.segment(spec, frame_s, CFG["segmentation"] | {"frame_percentile": 100})
    assert [(i.kind, i.start, i.end) for i in ivs] == [
        ("sound", 10, 120), ("silence", 120, 300), ("sound", 300, 350)]


# ---------------------------------------------------------------- distances, classification

def test_levenshtein():
    assert levenshtein(list("ABCCD"), list("ACCD")) == 1
    assert levenshtein(["UP", "SILENCE", "DOWN"], ["DOWN", "SILENCE", "UP"]) == 2


def test_cross_validation_separable():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    y = np.repeat([0, 1], 20)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)

    def method(train, test, y_train, seed):
        return classify.kmedoids_classifier(D[np.ix_(train, train)], D[np.ix_(test, train)],
                                            y_train, 2, seed)

    res = classify.cross_validate(method, y, [0, 1, 2, 3], 5)
    assert res.summary()["mean"] == 100.0
    assert classify.confusion(y, res.predictions, 2).sum() == 4 * len(y)


# ---------------------------------------------------------------- metrics, SNR

def test_entropy_rate_gap():
    rng = np.random.default_rng(0)
    seq = ["FLAT", "SILENCE"] * 10
    assert metrics.transition_entropy(seq) == 0.0
    assert metrics.entropy_rate_gap(seq, 200, rng) > 0
    assert np.isnan(metrics.entropy_rate_gap(["FLAT"], 200, rng))


def test_within_token_distance():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert metrics.within_token_distance(v)[0] == pytest.approx(0.0)


def test_noise_reaches_target_range():
    rng = np.random.default_rng(0)
    sr = 44100
    t = np.arange(sr) / sr
    x = np.zeros_like(t)
    x[sr // 4: sr // 2] = np.sin(2 * np.pi * 4000 * t[sr // 4: sr // 2])
    x += 1e-4 * rng.standard_normal(len(x))
    assert snr.snr_db(x) > 12
    for low, high in [(9, 12), (6, 9), (3, 6), (None, 2)]:
        _, achieved, _ = snr.add_noise_to_range(x, low, high, np.random.default_rng(1))
        assert (low is None or achieved >= low) and achieved <= high

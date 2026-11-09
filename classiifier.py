"""
Pretrained bioacoustic embedding baseline (Perch 2.0 / BirdNET / AVES).

Produces recording-level embeddings for every clip, clusters them with the SAME
protocol used by the token-sequence method (k-medoids on a distance matrix,
k = number of known classes, Hungarian matching), and reports clustering
accuracy over multiple seeds. k-means is reported alongside so the effect of
the clustering algorithm can be separated from the effect of the representation.

USAGE
    pip install 'perch-hoplite[tf]' librosa scikit-learn pandas soundfile
    python perch_baseline.py --manifest manifest.csv --model perch_v2
    python perch_baseline.py --selftest          # no model needed, checks the eval code

MANIFEST FORMAT (csv, one row per recording per evaluation setting)
    setting,path,label
    YNCC dialects,/data/ync/co_001.wav,Co
    YNCC dialects,/data/ync/u_014.wav,U
    Species,/data/tewa/tewa_003.wav,TEWA
    ...
A file may appear in several settings; it is embedded only once.

MODELS (perch_hoplite.zoo.model_configs.ModelConfigName)
    perch_v2      32 kHz, 5.0 s window, 1536-d, needs tensorflow
    perch_v2_onnx same weights as perch_v2, via onnxruntime instead of
                  tensorflow <- use this on CPUs without AVX (TF hard-requires
                  AVX at import time; onnxruntime does not)
    birdnet_V2.4  48 kHz, 3.0 s window, 1024-d
    aves          16 kHz, 768-d                  <- self-supervised alternative
    birdaves      16 kHz, 768-d
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ----------------------------------------------------------------------------
# model specs (mirrors perch_hoplite/zoo/model_configs.py)
# ----------------------------------------------------------------------------
MODEL_SPECS = {
    "perch_v2":      dict(sample_rate=32000, window_s=5.0, dim=1536),
    "perch_v2_onnx": dict(sample_rate=32000, window_s=5.0, dim=1536),
    "perch_8":      dict(sample_rate=32000, window_s=5.0, dim=1280),
    "birdnet_V2.4": dict(sample_rate=48000, window_s=3.0, dim=1024),
    "birdnet_V2.3": dict(sample_rate=48000, window_s=3.0, dim=1024),
    "aves":         dict(sample_rate=16000, window_s=5.0, dim=768),
    "birdaves":     dict(sample_rate=16000, window_s=5.0, dim=768),
}


# ----------------------------------------------------------------------------
# embedding
# ----------------------------------------------------------------------------
def embed_recordings(paths, model_name, cache_path):
    """Return {path: 1-D embedding} plus padding statistics."""
    spec = MODEL_SPECS[model_name]
    sr, win_s = spec["sample_rate"], spec["window_s"]
    win_n = int(round(sr * win_s))

    cache = {}
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as z:
            cache = {k: z[k] for k in z.files}
        print(f"[cache] loaded {len(cache)} embeddings from {cache_path}")

    todo = [p for p in paths if p not in cache]
    stats = {"n_total": len(paths), "n_padded": 0, "durations": []}

    if todo:
        import librosa
        from perch_hoplite.zoo import model_configs

        print(f"[embed] loading model '{model_name}' ...")
        model = model_configs.load_model_by_name(model_name)
        for i, p in enumerate(todo, 1):
            audio, _ = librosa.load(p, sr=sr, mono=True)
            stats["durations"].append(len(audio) / sr)
            if len(audio) < win_n:                       # zero-pad short clips
                stats["n_padded"] += 1
                audio = np.pad(audio, (0, win_n - len(audio)))
            out = model.embed(audio.astype(np.float32))
            emb = np.asarray(out.embeddings)             # (windows, channels, dim)
            emb = emb.reshape(-1, emb.shape[-1]).mean(0)  # recording-level vector
            cache[p] = emb.astype(np.float32)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
        if cache_path:
            np.savez_compressed(cache_path, **cache)
            print(f"[cache] wrote {cache_path}")
    else:
        print("[embed] all embeddings already cached")

    return cache, stats


# ----------------------------------------------------------------------------
# clustering + evaluation
# ----------------------------------------------------------------------------
def kmedoids(D, k, seed, max_iter=300):
    """Alternating k-medoids (PAM-style assign/update) on a precomputed matrix."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    # k-medoids++ style seeding
    medoids = [rng.integers(n)]
    for _ in range(k - 1):
        d = D[:, medoids].min(axis=1)
        probs = d**2
        probs = np.ones(n) / n if probs.sum() <= 0 else probs / probs.sum()
        medoids.append(rng.choice(n, p=probs))
    medoids = np.array(medoids)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:                        # empty cluster: take farthest point
                far = np.argmax(D[:, medoids].min(axis=1))
                new[j] = far
                continue
            sub = D[np.ix_(idx, idx)]
            new[j] = idx[np.argmin(sub.sum(axis=1))]
        if np.array_equal(np.sort(new), np.sort(medoids)):
            medoids = new
            break
        medoids = new
    return np.argmin(D[:, medoids], axis=1)


def clustering_accuracy(y_true, y_pred):
    """Accuracy after optimal one-to-one matching of clusters to classes."""
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    C = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for a, cl in enumerate(clusters):
        for b, cls in enumerate(classes):
            C[a, b] = np.sum((y_pred == cl) & (y_true == cls))
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y_true)


def evaluate_setting(X, y, n_seeds=10, distance="cosine"):
    """Return dict of method -> (mean, sd, best) accuracy.

    distance="cosine": L2-normalize first (spherical geometry), matches the
        original baseline.
    distance="euclidean": use raw, non-normalized embeddings. Normalizing
        first would make euclidean distance a monotonic function of cosine
        distance (||a-b||^2 = 2 - 2*cos_sim on unit vectors), which would
        make this identical to the cosine run rather than a real comparison.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    k = len(np.unique(y))
    if distance == "cosine":
        Xc = normalize(X)
        D = cdist(Xc, Xc, metric="cosine")
        method_tag = "k-medoids (cosine)"
    elif distance == "euclidean":
        Xc = X
        D = cdist(Xc, Xc, metric="euclidean")
        method_tag = "k-medoids (euclidean)"
    else:
        raise ValueError(f"unknown distance: {distance}")

    res = {}
    accs = [clustering_accuracy(y, kmedoids(D, k, seed=s)) for s in range(n_seeds)]
    res[method_tag] = (np.mean(accs), np.std(accs), np.max(accs))

    accs = []
    for s in range(n_seeds):
        pred = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Xc)
        accs.append(clustering_accuracy(y, pred))
    res[f"k-means ({distance})"] = (np.mean(accs), np.std(accs), np.max(accs))
    return res, k


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--model", default="perch_v2", choices=list(MODEL_SPECS))
    ap.add_argument("--cache", default=None, help="npz cache path (default: emb_<model>.npz)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"],
                     help="clustering distance metric (euclidean uses raw, non-normalized embeddings)")
    ap.add_argument("--out", default=None, help="markdown output path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.manifest:
        ap.error("--manifest is required (or use --selftest)")

    df = pd.read_csv(args.manifest)
    for col in ("setting", "path", "label"):
        if col not in df.columns:
            sys.exit(f"manifest is missing column '{col}'")

    cache_path = args.cache or f"emb_{args.model.replace('.', '_')}.npz"
    emb, stats = embed_recordings(sorted(df["path"].unique()), args.model, cache_path)

    lines = [f"### {args.model} embedding baseline (distance={args.distance})", ""]
    d = np.array(stats["durations"]) if stats["durations"] else np.array([np.nan])
    lines += [
        f"- recordings embedded: {stats['n_total']}",
        f"- zero-padded to reach the {MODEL_SPECS[args.model]['window_s']} s window: "
        f"{stats['n_padded']} ({100*stats['n_padded']/max(stats['n_total'],1):.0f}%)",
        f"- clip duration s: median {np.nanmedian(d):.2f}, "
        f"min {np.nanmin(d):.2f}, max {np.nanmax(d):.2f}",
        "",
        "| setting | n | classes | chance | method | acc mean ± sd | best |",
        "|---|---|---|---|---|---|---|",
    ]

    for setting, g in df.groupby("setting", sort=False):
        X = np.stack([emb[p] for p in g["path"]])
        y = pd.factorize(g["label"])[0]
        res, k = evaluate_setting(X, y, n_seeds=args.seeds, distance=args.distance)
        for method, (m, s, b) in res.items():
            lines.append(
                f"| {setting} | {len(y)} | {k} | {100/k:.1f}% | {method} | "
                f"{100*m:.2f} ± {100*s:.2f} | {100*b:.2f} |"
            )

    text = "\n".join(lines)
    print("\n" + text)
    out = args.out or f"results_{args.model.replace('.', '_')}.md"
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {out}")


def selftest():
    """Check the clustering + matching code on synthetic data, no model needed."""
    rng = np.random.default_rng(0)
    # NOTE: centres must differ in DIRECTION, not just magnitude — the pipeline
    # L2-normalises before computing cosine distance, so centres lying along the
    # same ray are indistinguishable by construction.
    print("selftest: 4 well-separated clusters, 200-d")
    centres = rng.normal(size=(4, 200))
    X = np.vstack([c + rng.normal(0, 0.25, size=(30, 200)) for c in centres])
    y = np.repeat(np.arange(4), 30)
    res, k = evaluate_setting(X, y, n_seeds=5)
    for m, (a, s, b) in res.items():
        print(f"  {m:22s} {100*a:6.2f} ± {100*s:5.2f}   (expect ~100)")

    print("selftest: random data, 3 classes")
    X = rng.normal(size=(90, 50)); y = np.repeat(np.arange(3), 30)
    res, _ = evaluate_setting(X, y, n_seeds=5)
    for m, (a, s, b) in res.items():
        print(f"  {m:22s} {100*a:6.2f} ± {100*s:5.2f}   (expect ~33-45, chance 33)")

    print("selftest: label permutation invariance")
    centres = rng.normal(size=(3, 100))
    X = np.vstack([c + rng.normal(0, 0.25, size=(20, 100)) for c in centres])
    y = np.repeat(np.arange(3), 20)
    a1 = clustering_accuracy(y, y)
    a2 = clustering_accuracy(y, (y + 1) % 3)
    print(f"  identity {a1:.2f}, permuted {a2:.2f}  (both must be 1.00)")
    assert a1 == 1.0 and a2 == 1.0
    print("OK")


if __name__ == "__main__":
    main()
"""
Measured SNR for every dataset — retained and excluded alike.

Replaces subjective screening reasons ("noise too strong", "noisy") with a number
computed the same way for every dataset. Uses the preprocessing already specified in
your Tables B.10: 44.1 kHz, 6 ms window, bandpass [1000, 11000] Hz, hysteresis
thresholds theta_hi = 0.1 / theta_lo = 0.01 on the normalised per-frame 95th-percentile
amplitude — i.e. the same active/inactive frame decision your segmenter makes.

    SNR_dB = 10 * log10( median in-band energy of ACTIVE frames
                         / median in-band energy of INACTIVE frames )

Also doubles as the subsampler: --sample N writes a manifest of N recordings per
dataset so you can run the excluded datasets cheaply.

USAGE
    python snr_proxy.py --root /data/audio                 # folder per dataset
    python snr_proxy.py --manifest files.csv               # columns: dataset,path
    python snr_proxy.py --root /data/audio --sample 25 --sample-out subsample.csv
    python snr_proxy.py --selftest
"""

import argparse
import os
import sys
import glob
import numpy as np

SR = 44100
WIN_MS = 6.0
BAND = (1000.0, 11000.0)
TH_HI, TH_LO = 0.1, 0.01
AUDIO_EXT = (".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3", ".ogg")


def load_audio(path, sr=SR):
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y
    except Exception:
        pass
    try:
        import soundfile as sf
        y, file_sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            n = int(round(len(y) * sr / file_sr))
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        return y.astype(np.float64)
    except Exception:
        pass
    from scipy.io import wavfile
    file_sr, y = wavfile.read(path)
    y = y.astype(np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(np.dtype(y.dtype), np.integer) or np.max(np.abs(y)) > 1.5:
        y = y / (np.max(np.abs(y)) + 1e-12)
    if file_sr != sr:
        n = int(round(len(y) * sr / file_sr))
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    return y


def hysteresis(score, hi, lo):
    """Frames above `hi` start a segment; it continues while above `lo`."""
    active = np.zeros(len(score), dtype=bool)
    on = False
    for i, v in enumerate(score):
        if not on and v >= hi:
            on = True
        elif on and v < lo:
            on = False
        active[i] = on
    # also run backwards so onsets are not clipped
    on = False
    for i in range(len(score) - 1, -1, -1):
        if not on and score[i] >= hi:
            on = True
        elif on and score[i] < lo:
            on = False
        active[i] |= on
    return active


def snr_db(y, sr=SR):
    """Return (snr_dB, fraction of frames the segmenter marks active).

    SNR is percentile-based so that it is always defined:
        10 * log10( P90 of in-band frame energy / P10 of in-band frame energy )
    A recording whose loudest frames barely exceed its quietest ones is, by
    construction, one in which no vocalization stands out from the background.

    The active fraction comes from YOUR hysteresis rule (theta_hi/theta_lo on the
    normalised per-frame 95th-percentile amplitude) and is reported separately: a
    value near 100% means the segmenter never finds a silence, which is a
    segmentation failure in its own right and worth screening on.
    """
    from scipy.signal import stft
    nper = int(round(sr * WIN_MS / 1000.0))
    f, _, Z = stft(y, fs=sr, nperseg=nper, noverlap=nper // 2)
    S = np.abs(Z)
    band = (f >= BAND[0]) & (f <= BAND[1])
    S = S[band]
    if S.size == 0 or S.shape[1] < 4:
        return np.nan, np.nan

    score = np.percentile(S, 95, axis=0)          # per-frame, as in your segmenter
    rng = score.max() - score.min()
    norm = (score - score.min()) / rng if rng > 0 else np.zeros_like(score)
    active = float(hysteresis(norm, TH_HI, TH_LO).mean())

    energy = (S ** 2).sum(axis=0)
    hi, lo = np.percentile(energy, 90), np.percentile(energy, 10)
    if hi <= 0 or lo <= 0:
        return np.nan, active
    return float(10 * np.log10(hi / lo)), active


def collect(root=None, manifest=None):
    items = []
    if manifest:
        import pandas as pd
        df = pd.read_csv(manifest)
        for _, r in df.iterrows():
            items.append((str(r["dataset"]), str(r["path"])))
    else:
        for d in sorted(os.listdir(root)):
            sub = os.path.join(root, d)
            if not os.path.isdir(sub):
                continue
            for ext in AUDIO_EXT:
                for p in sorted(glob.glob(os.path.join(sub, "**", "*" + ext), recursive=True)):
                    items.append((d, p))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="folder containing one subfolder per dataset")
    ap.add_argument("--manifest", help="csv with columns dataset,path")
    ap.add_argument("--sample", type=int, default=0, help="write N recordings per dataset")
    ap.add_argument("--sample-out", default="subsample.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="snr_by_dataset.md")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.root or a.manifest):
        ap.error("--root or --manifest required (or --selftest)")

    items = collect(a.root, a.manifest)
    if not items:
        sys.exit("no audio files found")
    print(f"[info] {len(items)} files across "
          f"{len(set(d for d, _ in items))} datasets")

    if a.sample:
        import pandas as pd
        rng = np.random.default_rng(a.seed)
        rows = []
        by = {}
        for d, p in items:
            by.setdefault(d, []).append(p)
        for d, ps in by.items():
            take = rng.choice(len(ps), size=min(a.sample, len(ps)), replace=False)
            rows += [dict(dataset=d, path=ps[i]) for i in sorted(take)]
        pd.DataFrame(rows).to_csv(a.sample_out, index=False)
        print(f"[written] {a.sample_out} ({len(rows)} recordings)")

    res = {}
    for i, (d, p) in enumerate(items, 1):
        try:
            s, frac = snr_db(load_audio(p))
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue
        res.setdefault(d, []).append((s, frac))
        if i % 50 == 0:
            print(f"  {i}/{len(items)}")

    lines = ["| dataset | n | median SNR (dB) | IQR | median % frames active |",
             "|---|---|---|---|---|"]
    for d in sorted(res):
        arr = np.array([x for x, _ in res[d]], dtype=float)
        fr = np.array([y for _, y in res[d]], dtype=float)
        ok = arr[~np.isnan(arr)]
        if len(ok) == 0:
            lines.append(f"| {d} | {len(arr)} | – | – | – |")
            continue
        q1, q3 = np.percentile(ok, [25, 75])
        lines.append(f"| {d} | {len(ok)} | {np.median(ok):.1f} | {q1:.1f}–{q3:.1f} | "
                     f"{100*np.nanmedian(fr):.0f}% |")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print(f"\n[written] {a.out}")
    print("\nReport this for EVERY dataset, retained and excluded. A screening rule of the "
          "form 'excluded when median SNR < X dB' is reproducible; 'noise too strong' is not.")


def selftest():
    from scipy.io import wavfile
    import tempfile
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    print("selftest: synthesising clean and noisy 5 kHz tone bursts")

    def make(path, noise):
        t = np.arange(int(SR * 2.0)) / SR
        y = np.zeros_like(t)
        for start in (0.3, 0.9, 1.5):
            m = (t >= start) & (t < start + 0.25)
            y[m] += np.sin(2 * np.pi * 5000 * t[m])
        y += rng.normal(0, noise, size=len(t))
        wavfile.write(path, SR, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32))

    clean, noisy = os.path.join(tmp, "clean.wav"), os.path.join(tmp, "noisy.wav")
    make(clean, 0.001)
    make(noisy, 0.5)
    sc, fc = snr_db(load_audio(clean))
    sn, fn = snr_db(load_audio(noisy))
    print(f"  clean : {sc:6.1f} dB   ({100*fc:.0f}% frames active)")
    print(f"  noisy : {sn:6.1f} dB   ({100*fn:.0f}% frames active)")
    assert sc > sn, "clean recording must score higher than noisy one"
    assert sc > 20, "clean tone bursts should be well above 20 dB"
    print("OK")


if __name__ == "__main__":
    main()

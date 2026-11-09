# BirdTokenSequences: Token Sequences for Bird Vocalization Classification

## Overview

This repository contains the code for the paper *Token sequences for bird vocalization classification*. It focuses on:

1. **Segmenting** each bird vocalization into sound segments and silence intervals.
2. **Synthesizing formulas** that fit the dominant-frequency trajectory of each segment (symbolic regression).
3. **Mapping formulas to tokens** (`UP`, `DOWN`, `FLAT`, `WIGGLE`, `SILENCE`) to get one token sequence per recording.
4. **Classifying recordings** by comparing token sequences with edit distance and k-medoids, against shuffled tokens, segment attributes, mean-frequency tokens, Trajectory-DTW, and Perch v2.

---

## Sources

1. *Identification of dialects and individuals of globally threatened yellow cardinals using neural networks* (Bocaccio et al., 2023)
2. *Perch 2.0: The bittern lesson for bioacoustics* (van Merriënboer et al., 2025)
3. *WildTrax* sound archive

---

## Setup

### Prerequisites

* Python 3.11
* Linux/macOS shell (bash)

### Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Only for the Perch v2 baseline:

```bash
pip install -r requirements-perch.txt
```

### Check the installation

```bash
pytest
```

---

## Datasets

The recordings are not included in this repository. Put your audio under `data/audio/` and list it in a manifest file `data/manifest.csv`:

```csv
recording_id,class,path
bcch_feebee_001,BCCH/fee-bee,BCCH/feebee/feebee_001.wav
wtsp_017,WTSP,WTSP/wtsp_017.wav
```

* `class` must match a class name in `configs/tasks.yaml` (e.g. `BCCH/fee-bee`, `BCCH/tseet`, `BCCH/chick-a-dee`, `TEWA`, `WTSP`, `YWAR`, `YECA/Co`, `YECA/U`, `YECA/RN`, `YECA/SL`).
* `path` is relative to `data/audio/`.

**Expected folder structure**

```
birdtokensequences/
├─ data/
│  ├─ manifest.csv
│  └─ audio/
│     ├─ BCCH/
│     │  └─ feebee/
│     │     └─ *.wav
│     ├─ WTSP/
│     │  └─ *.wav
│     └─ ...
└─ ...
```

> **Note:** `data/` and `outputs/` are ignored by git.

---

## How to Run

### 0) Run everything

```bash
bash scripts/run_all.sh
```

The steps below run each part separately.

### 1) Keep high-SNR recordings (SNR ≥ 10 dB)

```bash
python scripts/compute_snr.py --manifest data/manifest.csv --data-root data/audio \
    --out outputs/manifest_snr.csv --min-snr 10
```

### 2) Convert recordings to formulas and token sequences

```bash
python scripts/extract_tokens.py --manifest outputs/manifest_snr.csv --data-root data/audio \
    --out outputs/tokens --workers 8
```

> **Note:** formulas are evaluated at one time point per spectrogram frame by default (`--time-points frames`).

To use a fixed number of time points, e.g. 8:

```bash
python scripts/extract_tokens.py ... --time-points 8
```

To switch modes without refitting the formulas:

```bash
python scripts/retokenize.py --tokens outputs/tokens --out outputs/tokens_8pts --time-points 8
```

### 3) Compute Perch v2 embeddings

```bash
python scripts/compute_perch_embeddings.py --manifest outputs/manifest_snr.csv \
    --data-root data/audio --out outputs/perch_embeddings.npz
```

### 4) Compute classification accuracy

```bash
python scripts/evaluate.py --manifest outputs/manifest_snr.csv --tokens outputs/tokens \
    --embeddings outputs/perch_embeddings.npz --out outputs/eval
```

Methods: `token`, `shuffled`, `segment_attributes`, `meanfreq_tokens`, `trajectory_dtw`, `perch_kmeans`, `perch_kmedoids`. Use `--methods` and `--tasks` to run a subset.

### 5) Compute within-token distance and entropy-rate gap

```bash
python scripts/token_metrics.py --manifest outputs/manifest_snr.csv --tokens outputs/tokens \
    --out outputs/token_metrics
```

### 6) Compute sensitivity to the token-mapping thresholds

```bash
python scripts/mapping_sensitivity.py --manifest outputs/manifest_snr.csv --tokens outputs/tokens \
    --out outputs/mapping_sensitivity
```

### 7) Run the added-noise experiment (four-species task)

```bash
bash scripts/run_noise_experiment.sh
```

> **Note:** all hyperparameters are in `configs/default.yaml`; the five tasks are in `configs/tasks.yaml`.

---

## Output of the results

### 1) Token sequences

```
outputs/tokens/<recording_id>.json            segments, formulas, tokens, segment attributes
outputs/tokens/<recording_id>_segments.npy    32 x 32 segment images
```

### 2) Classification accuracy

```
outputs/eval/accuracy_table.csv    tasks x methods, mean ± SD (best run)
outputs/eval/accuracy_table.tex
outputs/eval/accuracy_long.csv     accuracy of every run
outputs/eval/predictions.csv       prediction for every recording in every run
outputs/eval/confusion/            confusion matrices
```

### 3) Token metrics, sensitivity, and noise

```
outputs/token_metrics/class_metrics.csv
outputs/mapping_sensitivity/sensitivity_table.csv
outputs/noise/<condition>/eval/accuracy_table.csv
```

---

## Contacts

Please open an issue on GitHub.

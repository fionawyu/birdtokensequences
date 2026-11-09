# BirdTokenSequences: Token Sequences for Bird Vocalization Classification

## Overview

This repository contains the code for the paper *Token sequences for bird vocalization classification*. It focuses on:

1. **Synthesizing formulas** for the dominant-frequency trajectory of each sound segment of a bird vocalization (genetic programming).
2. **Mapping formulas to tokens** (`UP`, `DOWN`, `FLAT`, `WIGGLE`, `SILENCE`) to get one token sequence per recording.
3. **Classifying recordings** by edit distance + k-medoids, and comparing with shuffled tokens, segment attributes, mean-frequency tokens, Trajectory-DTW, and Perch v2.

---

## Sources

1. *Identification of dialects and individuals of globally threatened yellow cardinals using neural networks* (Bocaccio et al., 2023)
2. *Perch 2.0: The bittern lesson for bioacoustics* (van Merriënboer et al., 2025)
3. *WildTrax* sound archive

---

## Setup

### Prerequisites

* Python 3.12
* A C++20-capable compiler (e.g., `g++`)
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

### Compile C++ extensions

Run this in the repository root:

```bash
bash scripts/build_cpp_extensions.sh
```

---

## Datasets

The recordings are not included in this repository. Create a folder named `datasets_link` in the project root and place your data inside.

**Expected folder structure**

```
birdtokensequences/
├─ datasets_link/
│  ├─ Chickadee/              # BCCH chick-a-dee
│  ├─ Tseet/                  # BCCH tseet
│  ├─ Feebee/                 # BCCH fee-bee
│  ├─ WTSP/
│  └─ step12_clean_selected/
│     ├─ TEWA/
│     └─ YWAR/
├─ Plots_YNCC*/               # Yellow Cardinal dialects (Co, U, RN, SL)
└─ ...
```

> **Note:** each `main_*_revised.py` has its `data_path` at the top of the file; change it if your data is somewhere else.

---

## How to Run

### 1) Convert spectrograms to formulae

One script per class; the argument is the random seed (default 47):

```bash
python main_chickeadee_revised.py 47
python main_TSEET_revised.py 47
python main_feebee_revised.py 47
python main_TEWA_revised.py 47
python main_WTSP_revised.py 47
python main_YWAR_revised.py 47
python main_YNCC_Co_revised.py 47
python main_YNCC_U_revised.py 47
python main_YNCC_RN_revised.py 47
python main_YNCC_SL_revised.py 47
```

> **Note:** the segmentation and frequency-band parameters of each class are at the top of its script.

Collect the output folders as `./1017_all/1017_plots47_<class>_1017`.

### 2) Generate token sequences

```bash
python mapString.py
```

```bash
python build_diagnosis_inputs_from_yml.py --root ./1017_all --out-dir ./diagnosis/chapter1_diagnosis
```

### 3) Compute segment attributes

```bash
python scripts/segment_attributes.py \
    --segments-csv diagnosis/chapter1_diagnosis/segments.csv \
    --out results_segment_attributes/segment_attributes.csv --verify
```

### 4) Compute Trajectory-DTW distances

```bash
python contour_dtw_baseline/extract_contours.py
python contour_dtw_baseline/compute_dtw.py
```

### 5) Compute Perch v2 embeddings

```bash
python compute_perch_embeddings.py <manifest.tsv> emb_perch_v2_onnx.npz
```

### 6) Compute classification accuracy (four runs of five-fold cross-validation)

```bash
python heldout_cv_evaluation/scripts/run_main_cv.py
python heldout_cv_evaluation/scripts/validate_cv.py
python heldout_cv_evaluation/scripts/make_outputs.py
python heldout_cv_evaluation/scripts/make_pooled_confusion.py
```

### 7) Compute mean-frequency tokens, within-token distance and entropy-rate gap

```bash
python scripts/run_chapter1_pitch_diagnostic.py
python scripts/plot_chapter1_sequence_structure.py
```

### 8) Compute sensitivity to the token-mapping thresholds

```bash
python mapping_rule_sensitivity/scripts/run_sensitivity.py
```

### 9) Run the added-noise experiment (four-species task)

```bash
python scripts/screening_sensitivity/compute_snr.py
python snr_noise_injection_experiment/scripts/build_denoised_baseline.py
python snr_noise_injection_experiment/scripts/build_full_run.py
python snr_noise_injection_experiment/scripts/score_full_run.py
python heldout_cv_evaluation/scripts/run_noise_cv.py
```

---

## Output of the results

### 1) Formulae

```
./1017_plots47_<class>_.../<region>/formulae/*_formulae.yml
```

### 2) Token sequences

```
./1017_all/1017_plots47_<class>_1017/west/labels/silent/*_decomp.yml
```

### 3) Classification accuracy

```
heldout_cv_evaluation/outputs/main_datasets/heldout_accuracy_summary.csv
heldout_cv_evaluation/outputs/tables/heldout_accuracy_table.csv
heldout_cv_evaluation/outputs/noise_conditions/heldout_accuracy_summary.csv
```

---

## Contacts

Please open an issue on GitHub.

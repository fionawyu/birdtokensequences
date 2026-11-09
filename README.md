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
│  ├─ Chickadee/west/*.wav              # BCCH chick-a-dee
│  ├─ Tseet/west/*.wav                  # BCCH tseet
│  ├─ Feebee/west/*.wav                 # BCCH fee-bee
│  ├─ WTSP/west/*.wav
│  └─ step12_clean_selected/
│     ├─ TEWA/west/*.wav
│     └─ YWAR/west/*.wav
├─ Plots_YNCC*/                         # Yellow Cardinal dialects (Co, U, RN, SL)
├─ yncc/yncc_curated_data_process/      # Yellow Cardinal spectrogram clips (.mat)
└─ ...
```

> **Note:** each `main_*_revised.py` sets its `data_path` and its segmentation and frequency-band parameters at the top of the file.

---

## How to Run

Run every command from the repository root.

### 1) Convert spectrograms to formulae

```bash
python main_chickeadee_revised.py
python main_TSEET_revised.py
python main_feebee_revised.py
python main_TEWA_revised.py
python main_WTSP_revised.py
python main_YWAR_revised.py
python main_YNCC_Co_revised.py
python main_YNCC_U_revised.py
python main_YNCC_RN_revised.py
python main_YNCC_SL_revised.py
```

Collect the output folders into `./1017_all/`:

```bash
bash scripts/collect_runs.sh
```

### 2) Map formulae to token sequences

```bash
for d in ./1017_all/1017_plots*_1017; do FML_RUN_DIR="$d" python mapString.py; done
```

### 3) Build segment and sequence tables

```bash
python build_diagnosis_inputs_from_yml.py --root ./1017_all --out-dir diagnosis/chapter1_diagnosis
python scripts/build_diagnosis_inputs_from_yml_mat_features.py --root ./1017_all --out-dir diagnosis/chapter1_diagnosis_matfeat
```

### 4) Mean-frequency tokens, within-token distance and entropy-rate gap

```bash
python scripts/run_chapter1_pitch_diagnostic.py
python scripts/plot_chapter1_sequence_structure.py
```

### 5) Segment attributes

```bash
python scripts/segment_attributes.py \
    --segments-csv diagnosis/chapter1_diagnosis/segments.csv \
    --out results_segment_attributes/segment_attributes.csv --verify
python scripts/run_formula_attribute_baseline.py
```

### 6) Trajectory-DTW

```bash
python mat_to_audio.py
python contour_dtw_baseline/build_yncc_stitched_audio.py
python contour_dtw_baseline/extract_contours.py
python contour_dtw_baseline/compute_dtw.py
```

### 7) Perch v2 embeddings

```bash
python build_perch_manifest.py
python classiifier.py --manifest manifest.csv --model perch_v2_onnx
```

### 8) Classification accuracy (four runs of five-fold cross-validation)

```bash
python heldout_cv_evaluation/scripts/run_main_cv.py
python heldout_cv_evaluation/scripts/validate_cv.py
python heldout_cv_evaluation/scripts/make_outputs.py
python heldout_cv_evaluation/scripts/make_pooled_confusion.py
```

### 9) Added-noise experiment (four-species task)

List the four-species recordings in `snr_noise_injection_experiment/data/recordings_65.csv` (columns `species,recording_name,file_path`), then:

```bash
python snr_noise_injection_experiment/scripts/measure_baseline_snr.py
python snr_noise_injection_experiment/scripts/build_denoised_baseline.py
python snr_noise_injection_experiment/scripts/build_pilot.py
python snr_noise_injection_experiment/scripts/build_pilot_v2_denoised.py
python snr_noise_injection_experiment/scripts/build_full_run.py
bash snr_noise_injection_experiment/run_fits.sh \
    snr_noise_injection_experiment/manifests/full_run_manifest.tsv \
    outputs/noise_injection_pilot_v2/results
python snr_noise_injection_experiment/scripts/score_full_run.py
python heldout_cv_evaluation/scripts/run_noise_cv.py
```

### 10) Sensitivity to the token-mapping thresholds

```bash
python mapping_rule_sensitivity/scripts/run_sensitivity.py
```

### 11) Recording SNR

```bash
python scripts/screening_sensitivity/compute_snr.py
```

---

## Output of the results

### 1) Formulae and token sequences

```
./1017_all/1017_plots*_<class>_1017/west/formulae/*_formulae.yml
./1017_all/1017_plots*_<class>_1017/west/labels/silent/*_decomp.yml
```

### 2) Classification accuracy

```
heldout_cv_evaluation/outputs/main_datasets/heldout_accuracy_summary.csv
heldout_cv_evaluation/outputs/tables/heldout_accuracy_table.csv
heldout_cv_evaluation/outputs/confusion_matrices/
```

### 3) Added noise

```
heldout_cv_evaluation/outputs/noise_conditions/heldout_accuracy_summary.csv
heldout_cv_evaluation/outputs/tables/heldout_noise_conditions_table.csv
```

---

## Contacts

Please open an issue on GitHub.

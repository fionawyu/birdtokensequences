# Token sequences for bird vocalization classification

Code for the paper *Token sequences for bird vocalization classification*.

Each vocalization is split into sound segments. A formula is fitted to the
dominant-frequency trajectory of every segment by symbolic regression, and each
formula is mapped to one of the tokens `UP`, `DOWN`, `FLAT` or `WIGGLE`.
Silence intervals become `SILENCE`. Recordings are classified by comparing
their token sequences with edit distance and k-medoids.

## Installation

```bash
conda create -n tokenseq python=3.11 -y && conda activate tokenseq
pip install -r requirements.txt
pip install -r requirements-perch.txt   # only for the Perch v2 baseline
pytest                                  # unit tests + an end-to-end test on synthetic audio
```

## Data

The recordings are not distributed with this repository. They come from
laboratory recordings of Black-capped Chickadees, the WildTrax sound archive
(TEWA, WTSP, YWAR), and the published Yellow Cardinal dialect recordings
(Bocaccio et al., 2023). Describe your recordings in a manifest CSV:

```csv
recording_id,class,path
bcch_feebee_001,BCCH/fee-bee,bcch/feebee_001.wav
wtsp_017,WTSP,wildtrax/wtsp_017.wav
```

`class` must match the class names in [`configs/tasks.yaml`](configs/tasks.yaml),
and `path` is relative to `--data-root`.

## Reproducing the experiments

[`scripts/run_all.sh`](scripts/run_all.sh) runs every step below in order:

```bash
MANIFEST=data/manifest.csv DATA_ROOT=data/audio OUT=outputs bash scripts/run_all.sh
```

| Step | Script | Paper |
|---|---|---|
| SNR and the high-SNR subset (SNR ≥ 10 dB) | `compute_snr.py` | Sec. 3.1, App. D.1 |
| Spectrograms, segmentation, symbolic regression, tokens, segment attributes | `extract_tokens.py` | Sec. 2.2–2.6 |
| Perch v2 embeddings | `compute_perch_embeddings.py` | Sec. 3.2.2 |
| Classification accuracy and confusion matrices | `evaluate.py` | Tables 2, 3, 10 |
| Within-token distance and entropy-rate gap | `token_metrics.py` | Sec. 3.4.5, Fig. 5, App. D.2–D.3 |
| Sensitivity to the mapping thresholds | `mapping_sensitivity.py` | App. C.3, Table 9 |
| Added Gaussian noise (four-species task) | `run_noise_experiment.sh` (uses `add_noise.py`) | Sec. 3.3.2, Tables 4, 11 |

Every method in `evaluate.py` uses the same splits: four runs of stratified
five-fold cross-validation, with seeds in `configs/default.yaml`. The methods are:

| Key | Method |
|---|---|
| `token` | Token + k-medoids |
| `shuffled` | Shuffled tokens + k-medoids |
| `segment_attributes` | Segment attributes (41-dim) + k-medoids |
| `meanfreq_tokens` | Mean-frequency tertiles in tokens + k-medoids |
| `trajectory_dtw` | Trajectory-DTW + k-medoids |
| `perch_kmeans`, `perch_kmedoids` | Perch v2 embeddings + k-means / k-medoids |

## Time points for token mapping

The token rules are applied to a formula evaluated at uniformly spaced time
points over the segment. There are two modes:

* `frames` (default, used in the paper): one point per spectrogram frame of the segment;
* an integer, e.g. `8`: a fixed number of points per segment.

```bash
python scripts/extract_tokens.py ... --time-points 8
python scripts/retokenize.py --tokens outputs/tokens --out outputs/tokens_8pts --time-points 8   # no refitting
```

## Configuration

All hyperparameters are in [`configs/default.yaml`](configs/default.yaml), each
annotated with the section or table that defines it.

## Repository layout

```
configs/    hyperparameters and task definitions
tokenseq/   library: audio, segmentation, formula, symbolic_regression, tokens,
            attributes, distances, classify, methods, metrics, snr, perch, pipeline
scripts/    command-line entry points (table above)
tests/      unit and end-to-end tests
```

## License

MIT, see [LICENSE](LICENSE).

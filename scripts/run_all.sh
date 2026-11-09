#!/usr/bin/env bash
# End-to-end reproduction of the experiments. Edit the three paths below.
#   MANIFEST   CSV with recording_id,class,path (see data/README.md)
#   DATA_ROOT  directory the `path` column is relative to
#   OUT        output directory
set -euo pipefail

MANIFEST=${MANIFEST:-data/manifest.csv}
DATA_ROOT=${DATA_ROOT:-data/audio}
OUT=${OUT:-outputs}
WORKERS=${WORKERS:-8}

cd "$(dirname "$0")/.."

# 1. High-SNR dataset (Sec. 3.1)
python scripts/compute_snr.py --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --out "$OUT/manifest_snr.csv" --min-snr 10

# 2. Token sequences (Sec. 2)
python scripts/extract_tokens.py --manifest "$OUT/manifest_snr.csv" --data-root "$DATA_ROOT" \
    --out "$OUT/tokens" --workers "$WORKERS"

# 3. Perch v2 embeddings (Sec. 3.2.2; needs requirements-perch.txt)
python scripts/compute_perch_embeddings.py --manifest "$OUT/manifest_snr.csv" \
    --data-root "$DATA_ROOT" --out "$OUT/perch_embeddings.npz"

# 4. Classification accuracy, Tables 2, 3 and 10
python scripts/evaluate.py --manifest "$OUT/manifest_snr.csv" --tokens "$OUT/tokens" \
    --embeddings "$OUT/perch_embeddings.npz" --out "$OUT/eval"

# 5. Within-token distance and entropy-rate gap, Figure 5
python scripts/token_metrics.py --manifest "$OUT/manifest_snr.csv" --tokens "$OUT/tokens" \
    --out "$OUT/token_metrics"

# 6. Mapping-threshold sensitivity, Table 9
python scripts/mapping_sensitivity.py --manifest "$OUT/manifest_snr.csv" --tokens "$OUT/tokens" \
    --out "$OUT/mapping_sensitivity"

# 7. Added noise, Tables 4 and 11
bash scripts/run_noise_experiment.sh

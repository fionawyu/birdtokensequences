#!/usr/bin/env bash
# Added-noise experiment on the four-species task (Sec. 3.3.2, Tables 4 and 11).
# Expects $OUT/manifest_snr.csv from scripts/run_all.sh (steps 1-3).
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/audio}
OUT=${OUT:-outputs}
WORKERS=${WORKERS:-8}
TIME_POINTS=${TIME_POINTS:-frames}
TASK=four_species

cd "$(dirname "$0")/.."

python scripts/add_noise.py --manifest "$OUT/manifest_snr.csv" --data-root "$DATA_ROOT" \
    --task "$TASK" --out "$OUT/noise"

for cond_dir in "$OUT"/noise/*/; do
    cond=$(basename "$cond_dir")
    python scripts/extract_tokens.py --manifest "$cond_dir/manifest.csv" \
        --out "$cond_dir/tokens" --time-points "$TIME_POINTS" --workers "$WORKERS"
    python scripts/compute_perch_embeddings.py --manifest "$cond_dir/manifest.csv" \
        --out "$cond_dir/perch_embeddings.npz"
    python scripts/evaluate.py --manifest "$cond_dir/manifest.csv" --tokens "$cond_dir/tokens" \
        --embeddings "$cond_dir/perch_embeddings.npz" --tasks "$TASK" --out "$cond_dir/eval"
    echo "== $cond done"
done

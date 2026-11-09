#!/usr/bin/env bash
# Fit formulas for every (recording, noise condition) row of a manifest.
# Same call as one Slurm array task, run sequentially (seed 47).
#   bash snr_noise_injection_experiment/run_fits.sh <manifest.tsv> <results_dir>
# Manifest columns (tab-delimited, header on line 1):
#   task_id species recording_name condition target_db achieved_db wav_path ...
set -euo pipefail

MANIFEST="${1:?usage: run_fits.sh <manifest.tsv> <results_dir>}"
RESULTS_DIR="${2:?usage: run_fits.sh <manifest.tsv> <results_dir>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r task_id species recording_name condition target_db achieved_db wav_path rest; do
  out_dir="${RESULTS_DIR}/by_task/${species}/${recording_name}/${condition}"
  log_dir="${RESULTS_DIR}/runtime_logs/by_task/${species}/${recording_name}/${condition}"
  echo "task ${task_id}: ${species} ${recording_name} ${condition}"
  python run_consistency_recording.py \
    --audio "$wav_path" \
    --species "$species" \
    --output-dir "$out_dir" \
    --runtime-log-dir "$log_dir" \
    --seeds 47
done

#!/usr/bin/env bash
# Move the output folder of each main_*_revised.py run into
# ./1017_all/1017_plots<seed>_<class>_1017, the layout the later steps read.
#   bash scripts/collect_runs.sh [seed]        (default seed: 47)
set -euo pipefail

seed="${1:-47}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p 1017_all

collect() {
  local pattern="$1"
  local name="$2"
  local matches=( $pattern )
  if [ ! -d "${matches[0]}" ]; then
    echo "[skip] no folder matches $pattern"
    return
  fi
  if [ "${#matches[@]}" -gt 1 ]; then
    echo "[error] several folders match $pattern: ${matches[*]}" >&2
    exit 1
  fi
  mv "${matches[0]}" "1017_all/1017_plots${seed}_${name}_1017"
  echo "${matches[0]} -> 1017_all/1017_plots${seed}_${name}_1017"
}

collect "1017_plots${seed}_Chickadee_1017_mindur*"        Chickadee
collect "1017_plots${seed}_Tseet_mindur*"                 Tseet
collect "1017_plots${seed}_feebee_mindur*"                feebee
collect "1017_plots${seed}_TEWA_minDur*"                  TEWA
collect "1017_plots${seed}_WTSP_1017_minDur*"             WTSP
collect "1017_plots${seed}_YWAR_1017_minDur*"             YWAR
collect "1115_final_plots${seed}_YNCC_Co_1017_*"          CoCurate
collect "1017_plots${seed}_YNCC_U_1017_*"                 UCurate
collect "1017_plots${seed}_YNCC_RN_enhanced_1017_*"       RNCurate
collect "1017_plots${seed}_YNCC_SL_1017_*"                SLCurate

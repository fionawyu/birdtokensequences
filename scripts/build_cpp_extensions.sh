#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

compile() {
  local output="$1"
  local source="$2"
  shift 2
  g++ -std=c++20 -fPIC -shared "$@" -o "$output" "$source"
}

compile extCheckRestrOnSubfmls.so cpp/extCheckRestrOnSubfmls.cpp
compile extGetHeadAndInputs.so cpp/extGetHeadAndInputs.cpp
compile extGetSubfmlByInd.so cpp/extGetSubfmlByInd.cpp
compile extParseEval.so cpp/extParseEval.cpp
compile extParseEvalTwoVar.so cpp/extParseEvalTwoVar.cpp
compile extEvalRPN.so cpp/extEvalRPN.cpp
compile extParseEvalBatch.so cpp/extParseEvalBatch.cpp
compile extParseEvalTwoVarBatch.so cpp/extParseEvalTwoVarBatch.cpp
compile extEvalPN.so cpp/extEvalPN.cpp -O2

echo "Built C++ shared libraries in $repo_root"

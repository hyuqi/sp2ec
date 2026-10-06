#!/usr/bin/env bash
# Five MMBench-Video question prompts; all algorithm settings use runner defaults.
# Optional environment: MODEL_PATH=/path/to/checkpoint DATA_ROOT=/path/to/datasets
set -euo pipefail
case "${BASH_SOURCE[0]}" in
  */*) cd -- "${BASH_SOURCE[0]%/*}" ;;
esac
exec "${PYTHON:-python}" -u run_experiments.py \
  --model qwen2.5-vl-7b \
  --dataset mmbench \
  --cossim --knapspec \
  "$@"

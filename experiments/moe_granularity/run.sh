#!/usr/bin/env bash
# Expert-granularity sweep: split each expert into m finer ones while scaling k
# by m, so the pool (E*is=8192) AND the active capacity (k*is=1024) both stay
# fixed. Only granularity changes. Plus a dense is=1024 active-width twin.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_granularity

# (is E k) at fixed pool=8192, active=1024 (12.5% sparse); m = 1,2,4,8.
triples=("1024 8 1" "512 16 2" "256 32 4" "128 64 8")

configs=()
for t in "${triples[@]}"; do
  read -r is e k <<< "$t"
  match=("$dir"/*-is"${is}"-e"${e}"-k"${k}".yaml)
  configs+=("${match[0]}")
done
configs+=("$dir"/qwen3-dense-*.yaml)   # dense baseline, same active width

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

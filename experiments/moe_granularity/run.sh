#!/usr/bin/env bash
# Expert-granularity sweep: split each expert into m finer ones while scaling k
# by m, so the pool (E*is=12288) AND the active capacity (k*is=1536) both stay
# fixed. Only granularity changes.
# Usage: nohup bash experiments/moe_granularity/run.sh > logs/moe_granularity.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_granularity

# (is E k) at fixed pool=12288, active=1536 (12.5% sparse); m = 1,2,4,8,16.
triples=("1536 8 1" "768 16 2" "384 32 4" "192 64 8" "96 128 16")

configs=()
for t in "${triples[@]}"; do
  read -r is e k <<< "$t"
  configs+=("$dir"/qwen3_183m_a51m_is"${is}"_e"${e}"_k"${k}".yaml)
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

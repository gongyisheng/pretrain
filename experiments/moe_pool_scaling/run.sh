#!/usr/bin/env bash
# Pool-scaling sweep: fix top-2 routing and per-expert size (is=128), grow the
# routed pool E. Active capacity (k*is=256) stays fixed, so FLOPs/token are
# constant; only the total expert pool (and total params) grows.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_pool_scaling

pool_experts=(8 16 32 64 128)

configs=()
for e in "${pool_experts[@]}"; do
  match=("$dir"/*-is128-e"${e}"-k2.yaml)
  configs+=("${match[0]}")
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

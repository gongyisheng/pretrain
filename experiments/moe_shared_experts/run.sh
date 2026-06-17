#!/usr/bin/env bash
# Shared-experts sweep: n_shared_experts x n_routed_experts_per_token.
# 4x3 grid = 12 runs.
# Config filenames embed param counts, so each (s, k) is matched by suffix.
set -euo pipefail
cd "$(dirname "$0")/../.."

shared_counts=(0 1 2 4)
routed_ks=(1 2 4)
dir=experiments/moe_shared_experts

configs=()
for s in "${shared_counts[@]}"; do
  for k in "${routed_ks[@]}"; do
    match=("$dir"/*-shared-"${s}"-routed-"${k}".yaml)
    configs+=("${match[0]}")
  done
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

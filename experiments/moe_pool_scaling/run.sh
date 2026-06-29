#!/usr/bin/env bash
# Pool-scaling sweep: fix top-8 routing and per-expert size (is=192), grow the
# routed pool E. Active capacity (k*is=1536) stays fixed, so FLOPs/token are
# constant; only the total expert pool (and total params) grows. E=64 is the
# 183m_a51m benchmark shared across the moe experiments.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_pool_scaling

pool_experts=(16 32 64 128 256)

configs=()
for e in "${pool_experts[@]}"; do
  match=("$dir"/*_is192_e"${e}"_k8.yaml)
  configs+=("${match[0]}")
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

#!/usr/bin/env bash
# Pool-scaling sweep: fix the active experts (2 shared + top-6 routed) and per-expert
# size (is=192), grow the routed pool E. Active capacity ((s+k)*is=1536) stays fixed,
# so FLOPs/token are constant; only the total expert pool (and total params) grows.
# Load balancing is aux-loss-free (expert_bias, rate 1e-3). E=64 is the 188m_a51m
# benchmark shared across the moe experiments.
# Usage: nohup bash experiments/moe_sparsity/run.sh > logs/moe_sparsity.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_sparsity

pool_experts=(16 32 64 128 256)

configs=()
for e in "${pool_experts[@]}"; do
  match=("$dir"/*_is192_e"${e}"_s2r6.yaml)
  configs+=("${match[0]}")
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

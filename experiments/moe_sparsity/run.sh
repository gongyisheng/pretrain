#!/usr/bin/env bash
# Pool-scaling sweep: fix active experts (2 shared + top-6 routed) and per-expert size, grow the routed pool E at constant FLOPs/token.
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

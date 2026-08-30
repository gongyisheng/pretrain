#!/usr/bin/env bash
# Ultra-sparse MoE, Muon arm: routed pool E=384..1024 at fixed active budget (2 shared + top-6).
# Usage: nohup bash experiments/moe_ultra_sparse/run_muon.sh > logs/moe_ultra_sparse_muon.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_ultra_sparse

configs=()
for e in 384 512 768 1024; do
  match=("$dir"/*_is192_e"${e}"_s2r6_muon.yaml)
  configs+=("${match[0]}")
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

echo "=== All moe_ultra_sparse Muon runs complete ==="

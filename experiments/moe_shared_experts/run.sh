#!/usr/bin/env bash
# Shared-experts split sweep at fixed active capacity (s + k = 8 active experts,
# active intermediate = 8*192 = 1536). Vary how the 8 active experts split between
# always-on shared and top-k routed. Config filenames embed the (s, k) suffix.
# Usage: nohup bash experiments/moe_shared_experts/run.sh > logs/moe_shared_experts.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_shared_experts

splits=("0 8" "1 7" "2 6" "3 5" "4 4" "5 3" "6 2")

configs=()
for sk in "${splits[@]}"; do
  read -r s k <<< "$sk"
  match=("$dir"/*_s"${s}"_r"${k}".yaml)
  configs+=("${match[0]}")
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

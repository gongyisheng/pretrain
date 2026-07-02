#!/usr/bin/env bash
# Expert-capacity-factor sweep on the qwen3_183m_a51m benchmark (E=64, k=8,
# is=192). Compares dynamic no-drop routing (capf=none) against fixed capacity
# at factors 1.0/1.25/1.5/2.0. Measures the speed <-> token-drop <-> loss tradeoff.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/moe_capacity_factor

factors=(none 1.0 1.25 1.5 2.0)

configs=()
for f in "${factors[@]}"; do
  configs+=("$dir"/qwen3_183m_a51m_capf_"${f}".yaml)
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

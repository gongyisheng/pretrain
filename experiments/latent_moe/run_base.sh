#!/usr/bin/env bash
# LatentMoE run_base: benchmark + compression only. Standard-MoE benchmark (qwen3
# 188m_a51m, no latent) plus latent ℓ ∈ {256,128,64} with the pool E=64 and top-k=6
# unchanged. Expect quality to drop as ℓ shrinks (paper: reducing d without more
# experts hurts). Load balancing aux-loss-free (expert_bias 1e-3); Muon lr 1e-3; 50k steps.
# Usage: nohup bash experiments/latent_moe/run_base.sh > logs/latent_moe_base.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/latent_moe

configs=("$dir/qwen3_188m_a51m_e64.yaml")   # benchmark, no latent
for l in 256 128 64; do
  configs+=("$dir"/qwen3_*_l"${l}"_e64.yaml) # compression only, E=64, k=6
done

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

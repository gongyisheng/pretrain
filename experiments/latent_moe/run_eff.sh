#!/usr/bin/env bash
# LatentMoE run_eff (ℓ-MoE_eff): reinvest the α=d/ℓ savings into a larger routed pool (N'=αN) with top-k fixed at 6, aux-loss-free, Muon lr 1e-3, 50k steps.
# Usage: nohup bash experiments/latent_moe/run_eff.sh > logs/latent_moe_eff.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/latent_moe

configs=(
  "$dir"/qwen3_*_l256_e128.yaml
  "$dir"/qwen3_*_l128_e256.yaml
  "$dir"/qwen3_*_l64_e512.yaml
)

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

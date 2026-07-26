#!/usr/bin/env bash
# LatentMoE run_acc (ℓ-MoE_acc, paper's recommended variant): reinvest into both the
# routed pool AND active experts. α-matched diagonal from the E=64 base, N' = αN and
# K' = αK (ℓ=256→E128/k12, ℓ=128→E256/k24, ℓ=64→E512/k48). Active params lifted back
# to ≈ the benchmark. Load balancing aux-loss-free (expert_bias 1e-3); Muon lr 1e-3; 50k steps.
# Usage: nohup bash experiments/latent_moe/run_acc.sh > logs/latent_moe_acc.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/latent_moe

configs=(
  "$dir"/qwen3_*_l256_e128_k12.yaml
  "$dir"/qwen3_*_l128_e256_k24.yaml
  "$dir"/qwen3_*_l64_e512_k48.yaml
)

for config in "${configs[@]}"; do
  echo "=== Training ${config} ==="
  uv run python scripts/train.py --config "${config}"
done

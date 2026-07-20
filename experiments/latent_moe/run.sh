#!/usr/bin/env bash
# LatentMoE (arXiv:2601.18089) sweep over the qwen3 188m_a51m benchmark
# (d=512, is=192, 2 shared + top-6 routed, E=64). Routed experts run in a compressed
# latent space ℓ (α = d/ℓ); intermediate width m is held fixed. Three groups:
#
#   run_base : benchmark + compression only. Latent ℓ ∈ {256,128,64}, pool E=64 and
#              top-k=6 unchanged. Expect quality to drop as ℓ shrinks (paper: reducing
#              d without more experts hurts).
#   run_eff  : ℓ-MoE_eff — reinvest the α savings into a larger routed pool, top-k
#              fixed. N' = αN (E=128/256/512 for ℓ=256/128/64). Active cost ≈ constant.
#   run_acc  : ℓ-MoE_acc — additionally scale active experts, K' = αK (k=12/24/48 for
#              ℓ=256/128/64), N' = αN. The paper's recommended accuracy variant.
#
# Both reinvest groups use the α-matched diagonal from the E=64 base. Load balancing is
# aux-loss-free (expert_bias, rate 1e-3); Muon lr 1e-3; 50k steps for every run.
# Usage:
#   nohup bash experiments/latent_moe/run.sh all  > logs/latent_moe.log 2>&1 &
#   bash experiments/latent_moe/run.sh base       # or: eff | acc | all
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=experiments/latent_moe

train() {
  for config in "$@"; do
    echo "=== Training ${config} ==="
    uv run python scripts/train.py --config "${config}"
  done
}

run_base() {
  local configs=("$dir/qwen3_188m_a51m_e64.yaml")   # benchmark, no latent
  for l in 256 128 64; do
    configs+=("$dir"/qwen3_*_l"${l}"_e64.yaml)       # compression only, E=64, k=6
  done
  train "${configs[@]}"
}

run_eff() {
  # α-matched diagonal, top-k fixed at 6
  train "$dir"/qwen3_*_l256_e128.yaml \
        "$dir"/qwen3_*_l128_e256.yaml \
        "$dir"/qwen3_*_l64_e512.yaml
}

run_acc() {
  # α-matched diagonal, top-k scaled: k = 6α (12/24/48)
  train "$dir"/qwen3_*_l256_e128_k12.yaml \
        "$dir"/qwen3_*_l128_e256_k24.yaml \
        "$dir"/qwen3_*_l64_e512_k48.yaml
}

case "${1:-all}" in
  base) run_base ;;
  eff)  run_eff ;;
  acc)  run_acc ;;
  all)  run_base; run_eff; run_acc ;;
  *) echo "usage: $0 [base|eff|acc|all]" >&2; exit 1 ;;
esac

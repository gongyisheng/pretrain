#!/bin/bash
# MoE router score-fn comparison: sigmoid vs softmax gating, both at aux_loss_coef=1e-3.
# Usage: nohup bash experiments/moe_router_score_fn/run.sh > logs/moe_router_score_fn.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    "qwen3_188m_a51m_softmax_aux1e-3"
    "qwen3_188m_a51m_sigmoid_aux1e-3"
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_router_score_fn/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_router_score_fn runs complete ==="

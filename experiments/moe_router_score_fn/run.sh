#!/bin/bash
# MoE router score-fn comparison: sigmoid gating swept over aux_loss_coef, with a
# single softmax reference point (aux1e-3). coef=0 keeps aux_loss on (MaxVio still
# logged) but applies no balancing pressure.
# Usage: nohup bash experiments/moe_router_score_fn/run.sh > logs/moe_router_score_fn.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

coefs=(1e-2 1e-3 1e-4 0)

configs=("qwen3_188m_a51m_softmax_aux1e-3")  # single softmax reference point
for coef in "${coefs[@]}"; do
    configs+=("qwen3_188m_a51m_sigmoid_aux${coef}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_router_score_fn/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_router_score_fn runs complete ==="

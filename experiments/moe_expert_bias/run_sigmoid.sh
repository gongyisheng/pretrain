#!/bin/bash
# MoE aux-loss-free expert-bias update-rate sweep, sigmoid router.
# Usage: nohup bash experiments/moe_expert_bias/run_sigmoid.sh > logs/moe_expert_bias_sigmoid.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

rates=(1e-5 1e-4 1e-3 1e-2)

configs=()

# Aux-loss baseline (same testbed) as a benchmark for loss-free expert bias.
configs+=("qwen3_188m_a51m_sigmoid_aux_coef1e-3")

for r in "${rates[@]}"; do
    configs+=("qwen3_188m_a51m_sigmoid_expert_bias_rate${r}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_expert_bias/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_expert_bias sigmoid runs complete ==="

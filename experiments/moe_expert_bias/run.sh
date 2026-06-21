#!/bin/bash
# MoE auxiliary-loss-free expert-bias update-rate sweep (find best update rate).
# Usage: nohup bash experiments/moe_expert_bias/run.sh > logs/moe_expert_bias.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

rates=(1e-4 1e-3 1e-2)

configs=()
for r in "${rates[@]}"; do
    configs+=("qwen3_133m_a35m_expert_bias_rate${r}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_expert_bias/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_expert_bias runs complete ==="

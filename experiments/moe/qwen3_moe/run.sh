#!/bin/bash
# Run MoE grid search experiments sequentially (lowest k first).
# Usage: nohup bash experiments/moe/qwen3_moe/run.sh > logs/moe_grid.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for config in qwen3_moe_233m_a45m qwen3_moe_233m_a57m qwen3_moe_233m_a82m qwen3_moe_233m_a133m qwen3_moe_233m_a233m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/moe/qwen3_moe/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All MoE grid search runs complete ==="

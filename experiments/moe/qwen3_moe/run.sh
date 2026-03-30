#!/bin/bash
# Run MoE grid search experiments sequentially (lowest k first).
# Usage: nohup bash experiments/moe/qwen3_moe/run.sh > logs/moe_grid.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for config in qwen3_moe_61m_a28m qwen3_moe_61m_a32m qwen3_moe_61m_a42m qwen3_moe_61m_a61m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/moe/qwen3_moe/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All MoE grid search runs complete ==="

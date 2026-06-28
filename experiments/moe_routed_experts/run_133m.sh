#!/bin/bash
# Run MoE 133M top-k grid search experiments sequentially (lowest k first).
# Usage: nohup bash experiments/moe_routed_experts/run_133m.sh > logs/moe_routed_133m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

PREFIX="qwen3"
TOTAL="133m"
ACTIVE=(34m 35m 39m 45m 57m 83m 133m)

for active in "${ACTIVE[@]}"; do
    config="${PREFIX}_${TOTAL}_a${active}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_routed_experts/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All MoE runs complete ==="

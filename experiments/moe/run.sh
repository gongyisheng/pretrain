#!/bin/bash
# Run MoE 506M top-k grid search experiments sequentially (lowest k first).
# Usage: nohup bash experiments/moe/run.sh > logs/moe_506m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

PREFIX="qwen3_moe"
TOTAL="506m"
ACTIVE=(109m 115m 128m 153m 204m 304m 506m)

for active in "${ACTIVE[@]}"; do
    config="${PREFIX}_${TOTAL}_a${active}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All MoE runs complete ==="

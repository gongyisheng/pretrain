#!/bin/bash
# Run all scaling law experiments sequentially (smallest first)
# Usage: nohup bash experiments/scaling_law/qwen3/run.sh > scaling_law_qwen3.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for config in qwen3_17m qwen3_33m qwen3_57m qwen3_145m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/scaling_law/qwen3/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All scaling law runs complete ==="

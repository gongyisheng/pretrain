#!/bin/bash
# Run FP8 vs bf16 experiments at two model sizes (57M, 0.5B) on Qwen3.
# Usage: nohup bash experiments/fp8/run.sh > logs/fp8.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for size in 57m 0.5b; do
    for dtype in bf16 fp8; do
        config="qwen3_${size}_${dtype}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/fp8/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== All FP8 runs complete ==="

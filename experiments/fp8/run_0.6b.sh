#!/bin/bash
# Run FP8 vs bf16 experiment at 0.5B on Qwen3.
# Usage: nohup bash experiments/fp8/run_0.6b.sh > logs/fp8_0.6b.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for dtype in bf16 fp8; do
    config="qwen3_0.5b_${dtype}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 0.5B FP8 runs complete ==="

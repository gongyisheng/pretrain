#!/bin/bash
# Run FP8 vs bf16 experiment at 57M on Qwen3.
# Usage: nohup bash experiments/fp8/run_57m.sh > logs/fp8_57m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for dtype in bf16 fp8; do
    config="qwen3_57m_${dtype}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 57M FP8 runs complete ==="

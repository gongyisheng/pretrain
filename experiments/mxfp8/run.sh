#!/bin/bash
# Run mxfp8 (all three GEMMs) vs bf16 experiment at 51M on Qwen3.
# Usage: nohup bash experiments/mxfp8/run.sh > logs/mxfp8.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_mxfp8)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mxfp8/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M mxfp8 runs complete ==="

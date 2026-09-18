#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: nohup bash experiments/int8_gradient_granularity/run.sh > logs/int8_gradient_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_int8_w8a8)
for granularity in rowwise blockwise1d_32 blockwise2d_32; do
    configs+=("qwen3_51m_int8_w8a8g8_grad_${granularity}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_gradient_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int8_gradient_granularity runs complete ==="

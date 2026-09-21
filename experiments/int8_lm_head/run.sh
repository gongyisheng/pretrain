#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: nohup bash experiments/int8_lm_head/run.sh > logs/int8_lm_head.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_77m_bf16)
for granularity in blockwise1d blockwise2d; do
    for block_size in 32; do
        configs+=("qwen3_77m_int8_w8a16_${granularity}_${block_size}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_lm_head/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== int8_lm_head runs complete ==="

#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: nohup bash experiments/int4_activation_granularity/run.sh > logs/int4_activation_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_int4_w4a16)
for layout in blockwise1d blockwise2d; do
    block_sizes=(16)
    if [[ "$layout" == blockwise1d ]]; then
        block_sizes=(16 32 64)
    fi
    for block_size in "${block_sizes[@]}"; do
        configs+=("qwen3_51m_int4_w4a4_act_${layout}_${block_size}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_activation_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_activation_granularity runs complete ==="

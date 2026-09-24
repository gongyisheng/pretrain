#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: nohup bash experiments/fp8_activation_granularity/run.sh > logs/fp8_activation_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_fp8_w8a16 qwen3_51m_fp8_w8a8_act_rowwise)
for block_size in 16 32 64 128; do
    configs+=("qwen3_51m_fp8_w8a8_act_blockwise1d_${block_size}")
done
configs+=(qwen3_51m_fp8_w8a8_act_blockwise1d_32_e5m2)
configs+=(qwen3_51m_fp8_w8a8_act_blockwise2d_32)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_activation_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_activation_granularity runs complete ==="

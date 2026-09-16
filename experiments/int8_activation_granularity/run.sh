#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: CUDA_VISIBLE_DEVICES=0 bash experiments/int8_activation_granularity/run.sh

set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a free GPU selected with nvidia-smi.}"

cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_int8_w8a16 qwen3_51m_int8_w8a8_act_rowwise)
for layout in blockwise1d blockwise2d; do
    configs+=("qwen3_51m_int8_w8a8_act_${layout}_32")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_activation_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int8_activation_granularity runs complete ==="

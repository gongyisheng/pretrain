#!/bin/bash
# Qwen3-51M BF16 baseline plus W8A16, W8A8, and W8A8G8 FP32/E8M0 pairs.
# Usage: nohup bash experiments/mxfp8_scale_dtype/run.sh > logs/mxfp8_scale_dtype_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

recipes=(w8a16 w8a8 w8a8g8)
scale_dtypes=(fp32 e8m0)

configs=()
configs+=(qwen3_51m_bf16)
for recipe in "${recipes[@]}"; do
    for scale_dtype in "${scale_dtypes[@]}"; do
        configs+=("qwen3_51m_fp8_${recipe}_scale_${scale_dtype}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mxfp8_scale_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M MXFP8 runs complete ==="

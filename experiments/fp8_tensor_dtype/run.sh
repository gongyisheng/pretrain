#!/bin/bash
# Run FP8 tensor-dtype cells at Qwen3 51M.
# Usage: nohup bash experiments/fp8_tensor_dtype/run.sh > logs/fp8_tensor_dtype.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_51m_bf16
    qwen3_51m_fp8_e4m3_w8a16
    qwen3_51m_fp8_e4m3_w16a8
    qwen3_51m_fp8_e4m3_w8a8
    qwen3_51m_fp8_e4m3_w8a8g8
    qwen3_51m_fp8_e4m3_w8a8_e5m2_g8
    qwen3_51m_fp8_e5m2_w8a16
    qwen3_51m_fp8_e5m2_w16a8
    qwen3_51m_fp8_e5m2_w8a8
    qwen3_51m_fp8_e5m2_w8a8g8
    qwen3_51m_fp8_e5m2_w8a8_e4m3_g8
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_tensor_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_tensor_dtype runs complete ==="

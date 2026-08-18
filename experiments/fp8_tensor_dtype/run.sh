#!/bin/bash
# Run FP8 tensor-dtype cells at Qwen3 51M.
# Usage: nohup bash experiments/fp8_tensor_dtype/run.sh > logs/fp8_tensor_dtype.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_51m_bf16
)

for dtype in e4m3 e5m2; do
    for tensor_dtypes in w8a16 w16a8 w8a8 w8a8g8; do
        configs+=("qwen3_51m_fp8_${dtype}_${tensor_dtypes}")
    done
    for gradient_dtype in e4m3 e5m2; do
        [[ "${gradient_dtype}" == "${dtype}" ]] && continue
        configs+=("qwen3_51m_fp8_${dtype}_w8a8_${gradient_dtype}_g8")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_tensor_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_tensor_dtype runs complete ==="

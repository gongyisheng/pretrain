#!/bin/bash
# Qwen3-51M BF16 baseline plus the W8A8 {G16, G8} x {FP32, E8M0 scale} sweep.
# Usage: nohup bash experiments/mxfp8/run.sh > logs/mxfp8_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

echo "=== qwen3_51m_bf16 ==="
echo "Started at: $(date)"
uv run python scripts/train.py --config experiments/mxfp8/qwen3_51m_bf16.yaml
echo "Finished at: $(date)"
echo ""

grad_formats=(g16 g8)
scale_dtypes=(fp32 e8m0)
for grad_format in "${grad_formats[@]}"; do
    for scale_dtype in "${scale_dtypes[@]}"; do
        if [[ "${scale_dtype}" == "e8m0" ]]; then
            config="qwen3_51m_mxfp8_w8a8${grad_format}"
        elif [[ "${grad_format}" == "g16" ]]; then
            config="qwen3_51m_fp8_e4m3_w8a8g16_blockwise1d_32"
        else
            config="qwen3_51m_fp8_e4m3_w8a8g8_blockwise1d_32"
        fi
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/mxfp8/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== 51M MXFP8 runs complete ==="

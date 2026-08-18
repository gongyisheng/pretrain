#!/bin/bash
# Run FP8 element-format (dtype) sweep vs bf16 at 51M on Qwen3, fixed tensorwise scaling.
# Usage: nohup bash experiments/fp8_dtype/run.sh > logs/fp8_dtype_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

# bf16 seeds 42/43/44 run first: their spread is the noise floor for every quant folder.
configs=(
    qwen3_51m_bf16
    qwen3_51m_bf16_seed43
    qwen3_51m_bf16_seed44
    qwen3_51m_fp8_std
    qwen3_51m_fp8_alle4m3
    qwen3_51m_fp8_alle5m2
    qwen3_51m_fp8_std_gwhp
    qwen3_51m_fp8_std_gihp
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_dtype runs complete ==="

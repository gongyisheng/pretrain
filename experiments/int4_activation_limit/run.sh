#!/bin/bash
# Qwen3-51M bf16 and unbounded W4A16 controls, plus activation limits 31, 15, 7, 3.
# All int4 runs use blockwise 1D (1, 16), fp32 block scales, and no global scale.
# Usage: nohup bash experiments/int4_activation_limit/run.sh > logs/int4_activation_limit_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for granularity in blockwise1d_16; do
    configs+=("qwen3_51m_int4_w4a16_${granularity}")
    for limit in 31 15 7 3; do
        configs+=("qwen3_51m_int4_w4a16_${granularity}_act_limit${limit}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_activation_limit/${config}.yaml" "$@"
    echo "Finished at: $(date)"
done

echo "=== 51M int4_activation_limit runs complete ==="

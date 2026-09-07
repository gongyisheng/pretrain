#!/bin/bash
# Run the int8 W8A8 activation-limit sweep (unbounded, 31, 15, 7, 3) across tensorwise and
# blockwise-2D 32x32 scaling, vs bf16 and a per-granularity int8 W8A16 reference, at 51M on Qwen3.
# Usage: nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for granularity in tensorwise blockwise2d_32; do
    configs+=("qwen3_51m_int8_w8a16_${granularity}" "qwen3_51m_int8_w8a8_${granularity}")
    for limit in 31 15 7 3; do
        configs+=("qwen3_51m_int8_w8a8_${granularity}_act_limit${limit}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_activation_limit/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== 51M int8_activation_limit runs complete ==="

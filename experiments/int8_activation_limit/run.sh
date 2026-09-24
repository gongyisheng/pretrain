#!/bin/bash
# Qwen3-51M bf16 baseline and W8A16/W8A8/W8A8G8 activation-limit sweep.
# Usage: nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for arm in w8a16 w8a8 w8a8g8; do
    for limit in 15 7 3 none; do
        configs+=("qwen3_51m_int8_${arm}_act_limit${limit}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_activation_limit/${config}.yaml" "$@"
    echo "Finished at: $(date)"
done

echo "=== 51M int8_activation_limit runs complete ==="

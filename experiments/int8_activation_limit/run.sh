#!/bin/bash
# Run the int8 W8A8 activation-limit sweep (unbounded, 31, 15, 7, 3, 2, 1) vs bf16 and int8 W8A16 at 51M on Qwen3.
# Usage: nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_int8_w8a16 qwen3_51m_int8_w8a8)
for limit in 31 15 7 3 2 1; do
    configs+=("qwen3_51m_int8_w8a8_act_limit${limit}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_activation_limit/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int8_activation_limit runs complete ==="

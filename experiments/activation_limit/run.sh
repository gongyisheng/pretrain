#!/bin/bash
# Run the bf16 activation-limit sweep (unbounded, 127, 63, 31, 15, 7, 3, 2, 1) at 51M on Qwen3.
# Usage: nohup bash experiments/activation_limit/run.sh > logs/activation_limit.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for limit in 127 63 31 15 7 3 2 1; do
    configs+=("qwen3_51m_act_limit${limit}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/activation_limit/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M activation_limit runs complete ==="

#!/bin/bash
# Compare bf16 with W4A16 quantization enabled after different update counts.
# Usage: nohup bash experiments/int4_weight_late_start/run.sh > logs/int4_weight_late_start.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
start_steps=(0 500 1000 1500 2000 2500 3000 5000 10000)
configs=()

for model in "${models[@]}"; do
    configs+=("${model}_bf16")
    for start_step in "${start_steps[@]}"; do
        configs+=("${model}_int4_w4a16_start_${start_step}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_late_start/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== int4_weight_late_start runs complete ==="

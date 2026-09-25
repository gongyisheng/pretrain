#!/bin/bash
# Compare bf16 with 16x16 int4 weight quantization, with and without Hadamard rotation.
# Usage: nohup bash experiments/int4_weight_hadamard/run.sh > logs/int4_weight_hadamard.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
block_sizes=(4 16 128)
configs=()

for model in "${models[@]}"; do
    configs+=("${model}_bf16" "${model}_int4_w4a16")
    for block_size in "${block_sizes[@]}"; do
        configs+=("${model}_int4_w4a16_hadamard_${block_size}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_hadamard/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== int4_weight_hadamard runs complete ==="

#!/bin/bash
# Fixed-token batch-size, learning-rate, and minimum-learning-rate sweep.
# Usage: bash experiments/batch_size/run.sh

set -e
cd "$(dirname "$0")/../.."

configs=()
for bs in 64 256 1024; do
    for lr in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
        coefficient="${lr%e-*}"
        exponent="${lr#*e-}"
        proportional_min_lr="${coefficient}e-$((exponent + 1))"
        previous_min_lr=""
        for min_lr in "${proportional_min_lr}" 1e-5; do
            if [[ "${min_lr}" == "${previous_min_lr}" ]]; then
                continue
            fi
            configs+=("qwen3_51m_bs_${bs}_lr_${lr}_min_lr_${min_lr}")
            previous_min_lr="${min_lr}"
        done
    done
done

echo "=== Running ${#configs[@]} configs ==="
for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
done

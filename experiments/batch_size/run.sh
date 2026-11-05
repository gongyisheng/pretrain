#!/bin/bash
# Fixed-token batch-size and learning-rate sweep.

set -e
cd "$(dirname "$0")/../.."

configs=()
for batch_size in 16k 64k 256k 1m; do
    for learning_rate in 2e-4 5e-4 1e-3 2e-3; do
        configs+=("qwen3_51m_bs_${batch_size}_lr${learning_rate}_token_13b")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
done

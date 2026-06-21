#!/bin/bash
# Run all lr_moe experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/lr_moe/run.sh > logs/lr_moe.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for lr in 1e-4 2e-4 3e-4 5e-4 1e-3 2e-3 3e-3 5e-3; do
    config="qwen3_133m_a35m_lr${lr}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lr_moe/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lr_moe runs complete ==="

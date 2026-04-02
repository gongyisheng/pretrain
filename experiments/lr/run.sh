#!/bin/bash
# Run all lr experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/lr/run.sh > logs/lr.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for lr in 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
    config="qwen3_57m_lr${lr}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lr/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lr runs complete ==="

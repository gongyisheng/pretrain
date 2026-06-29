#!/bin/bash
# Run all lr_moe experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/moe_lr/run.sh > logs/moe_lr.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for lr in 1e-4 2e-4 3e-4 5e-4 1e-3 2e-3 3e-3 5e-3; do
    configs+=("qwen3_183m_a51m_lr${lr}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_lr/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lr_moe runs complete ==="

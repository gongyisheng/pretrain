#!/bin/bash
# Run all lr_moe experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/moe_lr/run_a35m.sh > logs/moe_lr_a35m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for lr in 1e-4 2e-4 3e-4 5e-4 1e-3 2e-3 3e-3 5e-3; do
    configs+=("qwen3_133m_a35m_lr${lr}")
done
configs+=("qwen3_133m_a35m_lr5e-4_minlr_1e-4")

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_lr/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lr_moe runs complete ==="

#!/bin/bash
# Run all lr experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/lr/run.sh > logs/lr.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_lr1e-5 qwen3_57m_lr2e-5 qwen3_57m_lr5e-5 qwen3_57m_lr1e-4 qwen3_57m_lr2e-4 qwen3_57m_lr5e-4 qwen3_57m_lr1e-3; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lr/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lr runs complete ==="

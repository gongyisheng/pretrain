#!/bin/bash
# Run all dropout experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/dropout/run.sh > logs/dropout.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_drop0.0 qwen3_57m_drop0.01 qwen3_57m_drop0.02 qwen3_57m_drop0.05 qwen3_57m_drop0.1 qwen3_57m_drop0.2 qwen3_57m_drop0.5 qwen3_57m_drop0.9 qwen3_57m_drop0.95 qwen3_57m_drop0.99; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/dropout/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All dropout runs complete ==="

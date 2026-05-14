#!/bin/bash
# Run all warmup experiments sequentially (shortest to longest)
# Usage: nohup bash experiments/warmup/run.sh > logs/warmup.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

RATIOS="0.0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5"

echo "=== Warmup ratio sweep ==="
for ratio in $RATIOS; do
    config="qwen3_57m_warmup${ratio}"
    echo "--- ${config} --- Started at: $(date)"
    uv run python scripts/train.py --config "experiments/warmup/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== All warmup runs complete ==="

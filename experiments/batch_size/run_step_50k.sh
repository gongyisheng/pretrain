#!/bin/bash
# Run all batch_size experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

step=50k
for bs in 16k 64k 256k 1m; do
    config="qwen3_57m_bs_${bs}_step_${step}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All batch_size runs complete ==="

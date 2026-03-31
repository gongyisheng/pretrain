#!/bin/bash
# Run all batch_size experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_bs_16k qwen3_57m_bs_64k qwen3_57m_bs_256k qwen3_57m_bs_1m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All batch_size runs complete ==="

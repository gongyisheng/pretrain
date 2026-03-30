#!/bin/bash
# Run all batch_size experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_bs_8k qwen3_57m_bs_32k qwen3_57m_bs_128k qwen3_57m_bs_512k; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All batch_size runs complete ==="

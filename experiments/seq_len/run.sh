#!/bin/bash
# Run all seq_len experiments sequentially (shortest first)
# Usage: nohup bash experiments/seq_len/run.sh > logs/seq_len.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for seq_len in 512 1024 2048 4096 8192; do
    config="qwen3_57m_seq${seq_len}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/seq_len/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All seq_len runs complete ==="

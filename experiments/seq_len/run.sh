#!/bin/bash
# Run all seq_len experiments sequentially (shortest first)
# Usage: nohup bash experiments/seq_len/run.sh > logs/seq_len.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_seq512 qwen3_57m_seq1024 qwen3_57m_seq2048 qwen3_57m_seq4096 qwen3_57m_seq8192; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/seq_len/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All seq_len runs complete ==="

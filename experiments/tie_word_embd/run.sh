#!/bin/bash
# Run tie vs untie word embeddings experiment for Qwen3 57M
# Usage: nohup bash experiments/tie_word_embd/run.sh > logs/tie_word_embd.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for variant in tied untied; do
    config="qwen3_57m_${variant}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/tie_word_embd/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tie_word_embd runs complete ==="

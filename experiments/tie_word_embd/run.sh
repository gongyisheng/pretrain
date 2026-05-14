#!/bin/bash
# Run tie vs untie word embeddings experiment for Qwen3 57M and 0.5B
# Usage: nohup bash experiments/tie_word_embd/run.sh > logs/tie_word_embd.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in qwen3_57m_tied qwen3_57m_untied qwen3_0.5b_tied qwen3_0.5b_untied; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/tie_word_embd/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tie_word_embd runs complete ==="

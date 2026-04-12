#!/bin/bash
# Run tie vs untie word embeddings experiment for Qwen3 57M
# Usage: nohup bash experiments/tie_emb/run.sh > logs/tie_emb.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for variant in tied untied; do
    config="qwen3_57m_${variant}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/tie_emb/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tie_emb runs complete ==="

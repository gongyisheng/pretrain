#!/bin/bash
# Preprocess data for all vocab sizes (skip 50k — reuses existing data/)
# Usage: nohup bash experiments/tokenizer_vocab/preprocess_data.sh > logs/tokenizer_vocab_data.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for vocab in 10k 20k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "=== Preprocessing data: ${vocab} ==="
    echo "Started at: $(date)"
    uv run python scripts/preprocess_data.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All data preprocessed ==="

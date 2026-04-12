#!/bin/bash
# Train models for all vocab sizes
# Usage: nohup bash experiments/tokenizer_vocab/run.sh > logs/tokenizer_vocab.log 2>&1 &
# Prerequisites: run train_tokenizers.sh and preprocess_data.sh first

set -e
cd "$(dirname "$0")/../.."

for vocab in 10k 20k 50k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "=== Training model: vocab${vocab} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tokenizer_vocab runs complete ==="

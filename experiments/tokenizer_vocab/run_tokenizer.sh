#!/bin/bash
# Train tokenizers and preprocess data for all vocab sizes
# (skip 50k — reuses existing tokenizers/custom_bpe_50k and data/)
# Usage: nohup bash experiments/tokenizer_vocab/run_tokenizer.sh > logs/tokenizer_vocab_prep.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

echo "=== Step 1: Train tokenizers ==="
for vocab in 10k 20k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Training tokenizer: ${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/train_tokenizer.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Step 2: Preprocess data ==="
for vocab in 10k 20k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Preprocessing data: ${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/preprocess_data.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tokenizers trained and data preprocessed ==="

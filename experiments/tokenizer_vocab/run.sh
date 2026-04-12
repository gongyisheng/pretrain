#!/bin/bash
# Run tokenizer vocab size sweep experiment for Qwen3 57M
# Usage: nohup bash experiments/tokenizer_vocab/run.sh > logs/tokenizer_vocab.log 2>&1 &
#
# Step 1: train tokenizers (skip 50k — reuses existing tokenizers/custom_bpe_50k)
# Step 2: preprocess data for each vocab size (skip 50k — reuses existing data/)
# Step 3: train models

set -e
cd "$(dirname "$0")/../.."

VOCABS="10k 20k 100k 200k"  # 50k skipped — tokenizer and data already exist

echo "=== Step 1: Train tokenizers ==="
for vocab in $VOCABS; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Training tokenizer: ${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/train_tokenizer.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Step 2: Preprocess data ==="
for vocab in $VOCABS; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Preprocessing data: ${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/preprocess_data.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Step 3: Train models ==="
for vocab in 10k 20k 50k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Training model: vocab${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tokenizer_vocab runs complete ==="

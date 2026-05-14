#!/bin/bash
# Train tokenizers for all vocab sizes
# (50k tokenizer already exists at tokenizers/custom_bpe_50k — only train others)
# Usage: nohup bash experiments/tokenizer_vocab/run_tokenizer.sh > logs/tokenizer_vocab.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

echo "=== Train tokenizers ==="
for vocab in 10k 20k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "--- Training tokenizer: ${vocab} ---"
    echo "Started at: $(date)"
    uv run python scripts/train_tokenizer.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tokenizers trained ==="

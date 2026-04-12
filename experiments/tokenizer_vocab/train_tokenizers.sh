#!/bin/bash
# Train BPE tokenizers for all vocab sizes (skip 50k — reuses existing tokenizers/custom_bpe_50k)
# Usage: nohup bash experiments/tokenizer_vocab/train_tokenizers.sh > logs/tokenizer_vocab_tok.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for vocab in 10k 20k 100k 200k; do
    config="experiments/tokenizer_vocab/qwen3_57m_vocab${vocab}.yaml"
    echo "=== Training tokenizer: ${vocab} ==="
    echo "Started at: $(date)"
    uv run python scripts/train_tokenizer.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All tokenizers trained ==="

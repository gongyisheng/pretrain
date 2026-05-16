#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for name in bpe_200k superbpe_200k_t80k superbpe_200k_t160k superbpe_200k_t180k; do
    echo "==> Evaluating $name"
    uv run python scripts/eval_tokenizer.py \
        --tokenizer "tokenizers/experiments/$name" \
        --dataset openwebtext --split train --num_samples 10000
done

#!/usr/bin/env bash
# Evaluate bytes/token for the BPE baseline + every SuperBPE transition point.
# Usage: bash experiments/superbpe/run_eval.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

names=(bpe_200k)
for t in 20 40 60 80 100 120 140 160 180; do
    names+=("superbpe_200k_t${t}k")
done

for name in "${names[@]}"; do
    echo "==> Evaluating $name"
    uv run python scripts/eval_tokenizer.py \
        --tokenizer "tokenizers/experiments/$name" \
        --dataset openwebtext --split train --num_samples 10000
done

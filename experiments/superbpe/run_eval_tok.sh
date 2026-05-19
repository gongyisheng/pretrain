#!/usr/bin/env bash
# Evaluate bytes/token for every tokenizer trained by run_train.sh.
# Per-run JSON output is captured to logs/superbpe/eval/<name>.log
# (one JSON blob per file). Run collect_results.py afterward to
# aggregate into experiments/superbpe/results.csv.
#
# Usage: bash experiments/superbpe/run_eval_tok.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/superbpe/eval

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-10000}"

for V in 50 100 150 200; do
    folder="experiments/superbpe/v${V}k"

    # BPE baseline first.
    name="bpe_v${V}k"
    tokenizer="tokenizers/experiments/${name}"
    if [[ -d "$tokenizer" ]]; then
        echo "==> Evaluating ${name}"
        uv run python scripts/eval_tokenizer.py \
            --tokenizer "$tokenizer" \
            --dataset openwebtext --split train \
            --num_samples "$EVAL_NUM_SAMPLES" \
            2>&1 | tee "logs/superbpe/eval/${name}.log"
    else
        echo "skip: $tokenizer not trained yet"
    fi

    for cfg in "${folder}"/superbpe_*.yaml; do
        name="$(basename "$cfg" .yaml)"
        tokenizer="tokenizers/experiments/${name}"
        if [[ ! -d "$tokenizer" ]]; then
            echo "skip: $tokenizer not trained yet"
            continue
        fi
        echo "==> Evaluating ${name}"
        uv run python scripts/eval_tokenizer.py \
            --tokenizer "$tokenizer" \
            --dataset openwebtext --split train \
            --num_samples "$EVAL_NUM_SAMPLES" \
            2>&1 | tee "logs/superbpe/eval/${name}.log"
    done
done

echo "Done. Run: uv run python experiments/superbpe/collect_results.py"

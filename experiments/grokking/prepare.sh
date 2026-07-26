#!/usr/bin/env bash
# Idempotent data prep for grokking: builds the shared tokenizer and per-op data if missing.
# Usage: bash experiments/grokking/prepare.sh [op ...]
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

P=97
TRAIN_FRAC=0.3
TOKENIZER_DIR="tokenizers/grokking"
TOKENIZER_FILE="${TOKENIZER_DIR}/tokenizer.json"

if [ $# -ge 1 ]; then
    OPS=("$@")
else
    OPS=(add sub mul div)
fi

prep_tokenizer() {
    if [ -f "$TOKENIZER_FILE" ]; then
        echo "[prepare.sh] tokenizer exists: $TOKENIZER_FILE (skip)"
    else
        echo "[prepare.sh] building tokenizer → $TOKENIZER_FILE"
        uv run python experiments/grokking/generate_tokenizer.py --out_dir "$TOKENIZER_DIR"
    fi
}

prep_data() {
    local op="$1"
    local data_dir="data/grokking_${op}_p${P}_f${TRAIN_FRAC}"

    if [ -f "${data_dir}/train.bin" ] && [ -f "${data_dir}/val.bin" ] && \
       [ -f "${data_dir}/train_targets.bin" ] && [ -f "${data_dir}/val_targets.bin" ]; then
        echo "[prepare.sh] tokenized data exists: $data_dir (skip)"
        return
    fi

    if [ ! -f "${data_dir}/train_text.parquet" ] || [ ! -f "${data_dir}/val_text.parquet" ]; then
        echo "[prepare.sh] generating raw data for op=$op → $data_dir"
        uv run python experiments/grokking/generate_data.py \
            --op "$op" --p "$P" --train_frac "$TRAIN_FRAC"
    else
        echo "[prepare.sh] raw parquet exists for $op (skip generate_data)"
    fi

    echo "[prepare.sh] tokenizing $op → ${data_dir}/{train,val}.bin"
    uv run python experiments/grokking/tokenize_data.py \
        --data_dir "$data_dir" --tokenizer_path "$TOKENIZER_DIR"
}

prep_tokenizer
for op in "${OPS[@]}"; do
    prep_data "$op"
done

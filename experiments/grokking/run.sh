#!/usr/bin/env bash
# Run the full grokking sweep: 4 ops × 3 weight-decay values = 12 runs.
#
# Idempotently prepares the tokenizer and per-op data (skips any stage whose
# output already exists), then launches training.
#
# Usage:
#   # full 12-config sweep, sequential (one run at a time)
#   nohup bash experiments/grokking/run.sh > logs/grokking.log 2>&1 &
#
#   # full sweep with 4 concurrent runs sharing the visible GPU
#   MAX_CONCURRENCY=4 nohup bash experiments/grokking/run.sh > logs/grokking.log 2>&1 &
#
#   # single config (op=add, wd=1.0); MAX_CONCURRENCY ignored here
#   nohup bash experiments/grokking/run.sh add 1.0 > logs/grokking.log 2>&1 &
#
#   # pin to cuda:1 by prefixing CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run.sh > logs/grokking.log 2>&1 &
#
# MAX_CONCURRENCY (default 6) controls how many training runs execute in
# parallel during the sweep. Each concurrent run gets its own per-config log
# under logs/grokking/<op>_wd<wd>.log so output isn't interleaved. All runs
# share whatever GPUs CUDA_VISIBLE_DEVICES exposes — for the 1M-param grokking
# model this fits ~dozens of runs per GPU; bump cautiously.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# OPS=(add sub mul div)
OPS=(mul div)
WDS=(0.0 0.1 1.0)
P=97
TRAIN_FRAC=0.3
TOKENIZER_DIR="tokenizers/grokking"
TOKENIZER_FILE="${TOKENIZER_DIR}/tokenizer.json"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
PER_RUN_LOG_DIR="logs/grokking"

prep_tokenizer() {
    if [ -f "$TOKENIZER_FILE" ]; then
        echo "[run.sh] tokenizer exists: $TOKENIZER_FILE (skip)"
    else
        echo "[run.sh] building tokenizer → $TOKENIZER_FILE"
        uv run python experiments/grokking/generate_tokenizer.py --out_dir "$TOKENIZER_DIR"
    fi
}

prep_data() {
    local op="$1"
    local data_dir="data/grokking_${op}_p${P}_f${TRAIN_FRAC}"

    if [ -f "${data_dir}/train.bin" ] && [ -f "${data_dir}/val.bin" ] && \
       [ -f "${data_dir}/train_targets.bin" ] && [ -f "${data_dir}/val_targets.bin" ]; then
        echo "[run.sh] tokenized data exists: $data_dir (skip)"
        return
    fi

    if [ ! -f "${data_dir}/train_text.parquet" ] || [ ! -f "${data_dir}/val_text.parquet" ]; then
        echo "[run.sh] generating raw data for op=$op → $data_dir"
        uv run python experiments/grokking/generate_data.py \
            --op "$op" --p "$P" --train_frac "$TRAIN_FRAC"
    else
        echo "[run.sh] raw parquet exists for $op (skip generate_data)"
    fi

    echo "[run.sh] tokenizing $op → ${data_dir}/{train,val}.bin"
    uv run python experiments/grokking/tokenize_data.py \
        --data_dir "$data_dir" --tokenizer_path "$TOKENIZER_DIR"
}

run_one() {
    local op="$1"
    local wd="$2"
    local cfg="experiments/grokking/qwen3_1m_${op}_wd${wd}.yaml"
    echo "[run.sh] $cfg"
    uv run python scripts/train.py --config "$cfg"
}

run_one_logged() {
    # Background-friendly variant: redirects stdout/stderr to a per-config
    # log file so parallel jobs don't interleave output on the parent stream.
    local op="$1"
    local wd="$2"
    local cfg="experiments/grokking/qwen3_1m_${op}_wd${wd}.yaml"
    local log="${PER_RUN_LOG_DIR}/${op}_wd${wd}.log"
    echo "[run.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

prep_tokenizer

if [ $# -eq 2 ]; then
    prep_data "$1"
    run_one "$1" "$2"
else
    for op in "${OPS[@]}"; do
        prep_data "$op"
    done

    mkdir -p "$PER_RUN_LOG_DIR"
    echo "[run.sh] launching sweep with MAX_CONCURRENCY=$MAX_CONCURRENCY"
    running=0
    for op in "${OPS[@]}"; do
        for wd in "${WDS[@]}"; do
            run_one_logged "$op" "$wd" &
            running=$((running + 1))
            if [ "$running" -ge "$MAX_CONCURRENCY" ]; then
                wait -n
                running=$((running - 1))
            fi
        done
    done
    wait
fi

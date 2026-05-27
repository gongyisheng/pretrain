#!/usr/bin/env bash
# Run the full grokking weight-decay sweep: 4 ops × 3 weight-decay values = 12 runs.
#
# Configs live in experiments/grokking/weight_decay/.
#
# Idempotently prepares the tokenizer and per-op data (skips any stage whose
# output already exists), then launches training.
#
# Usage:
#   # full 12-config sweep, sequential (one run at a time)
#   nohup bash experiments/grokking/run_weight_decay.sh > logs/grokking_wd.log 2>&1 &
#
#   # full sweep with 6 concurrent runs sharing the visible GPU
#   MAX_CONCURRENCY=6 nohup bash experiments/grokking/run_weight_decay.sh > logs/grokking_wd.log 2>&1 &
#
#   # single config (op=add, wd=1.0); MAX_CONCURRENCY ignored here
#   nohup bash experiments/grokking/run_weight_decay.sh add 1.0 > logs/grokking_wd.log 2>&1 &
#
#   # pin to cuda:1 by prefixing CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run_weight_decay.sh > logs/grokking_wd.log 2>&1 &
#
# MAX_CONCURRENCY (default 6) controls how many training runs execute in
# parallel during the sweep. Each concurrent run gets its own per-config log
# under logs/grokking/<op>_wd<wd>.log so output isn't interleaved. All runs
# share whatever GPUs CUDA_VISIBLE_DEVICES exposes — for the 1M-param grokking
# model this fits ~dozens of runs per GPU; bump cautiously.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OPS=(add sub mul div)
WDS=(0.0 0.1 1.0)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
PER_RUN_LOG_DIR="logs/grokking"

run_one() {
    local op="$1"
    local wd="$2"
    local cfg="experiments/grokking/weight_decay/qwen3_1m_${op}_wd${wd}.yaml"
    echo "[run_weight_decay.sh] $cfg"
    uv run python scripts/train.py --config "$cfg"
}

run_one_logged() {
    # Background-friendly variant: redirects stdout/stderr to a per-config
    # log file so parallel jobs don't interleave output on the parent stream.
    local op="$1"
    local wd="$2"
    local cfg="experiments/grokking/weight_decay/qwen3_1m_${op}_wd${wd}.yaml"
    local log="${PER_RUN_LOG_DIR}/${op}_wd${wd}.log"
    echo "[run_weight_decay.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 2 ]; then
    bash experiments/grokking/prepare.sh "$1"
    run_one "$1" "$2"
else
    bash experiments/grokking/prepare.sh "${OPS[@]}"

    mkdir -p "$PER_RUN_LOG_DIR"
    echo "[run_weight_decay.sh] launching sweep with MAX_CONCURRENCY=$MAX_CONCURRENCY"
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

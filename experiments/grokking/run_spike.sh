#!/usr/bin/env bash
# Slingshot-spike ablation for grokking (sub task).
#
# Tests whether the paper's fp64 CE fix (Liu et al. 2025, arXiv:2605.06152)
# actually eliminates slingshot spikes in our setup, by isolating the
# weight-decay variable:
#
#   wd0.0_ce       - fp32 CE,  wd=0   (paper's regime; expected: spikes)
#   wd0.0_ce_fp64  - fp64 CE,  wd=0   (paper's fix;    expected: no spikes)
#   wd0.1_ce_fp64  - fp64 CE,  wd=0.1 (our regime;     observed: spikes)
#
# Configs live in experiments/grokking/spike/.
#
# Reuses the tokenizer and sub-task data prepared by run_weight_decay.sh; if
# either is missing, run experiments/grokking/run_weight_decay.sh first to
# build them.
#
# Usage:
#   # all 3 variants in parallel
#   nohup bash experiments/grokking/run_spike.sh > logs/grokking_spike.log 2>&1 &
#
#   # single variant
#   nohup bash experiments/grokking/run_spike.sh wd0.0_ce_fp64 > logs/grokking_spike.log 2>&1 &
#
#   # pin to a GPU
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run_spike.sh > logs/grokking_spike.log 2>&1 &
#
# MAX_CONCURRENCY (default 3) controls parallelism. Each run gets its own
# per-config log under logs/grokking/spike_<variant>.log.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(wd0.0_ce wd0.0_ce_fp64 wd0.1_ce_fp64)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
PER_RUN_LOG_DIR="logs/grokking"
TOKENIZER_FILE="tokenizers/grokking/tokenizer.json"
DATA_DIR="data/grokking_sub_p97_f0.3"

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "[run_spike.sh] tokenizer missing: $TOKENIZER_FILE"
    echo "  → run experiments/grokking/run_weight_decay.sh first to build tokenizer + data"
    exit 1
fi

if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ]; then
    echo "[run_spike.sh] tokenized sub data missing under $DATA_DIR"
    echo "  → run experiments/grokking/run_weight_decay.sh first to build data"
    exit 1
fi

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/spike/qwen3_1m_sub_${variant}.yaml"
    local log="${PER_RUN_LOG_DIR}/spike_${variant}.log"
    echo "[run_spike.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 1 ]; then
    run_one "$1"
    exit 0
fi

echo "[run_spike.sh] launching spike ablation with MAX_CONCURRENCY=$MAX_CONCURRENCY"
running=0
for variant in "${VARIANTS[@]}"; do
    run_one "$variant" &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_CONCURRENCY" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

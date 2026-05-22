#!/usr/bin/env bash
# Run the full grokking sweep: 4 ops × 3 weight-decay values = 12 runs.
#
# Prerequisites:
#   1. tokenizers/grokking/tokenizer.json   (run experiments/grokking/generate_tokenizer.py)
#   2. data/grokking_{add,sub,mul,div}_p97_f0.3/  (run generate_data.py per op)
#
# Usage:
#   bash experiments/grokking/run.sh           # run all 12 sequentially
#   bash experiments/grokking/run.sh add 1.0   # run a single config
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OPS=(add sub mul div)
WDS=(0.0 0.1 1.0)

run_one() {
    local op="$1"
    local wd="$2"
    local cfg="experiments/grokking/qwen3_1m_${op}_wd${wd}.yaml"
    echo "[run.sh] $cfg"
    uv run python scripts/train.py --config "$cfg"
}

if [ $# -eq 2 ]; then
    run_one "$1" "$2"
else
    for op in "${OPS[@]}"; do
        for wd in "${WDS[@]}"; do
            run_one "$op" "$wd"
        done
    done
fi

#!/usr/bin/env bash
# Train the SuperBPE grid sweep: 134 configs (4 BPE baselines + 130 SuperBPE).
# Smallest V first, BPE baseline before SuperBPE configs in each folder.
# Each run streams to logs/superbpe/train/<name>.log.
#
# Usage: nohup bash experiments/superbpe/run_train.sh > logs/superbpe_train.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."


for V in 50 100 150 200; do
    folder="experiments/superbpe/v${V}k"
    if [[ ! -d "$folder" ]]; then
        echo "error: $folder missing" >&2
        exit 1
    fi

    # BPE baseline
    baseline="${folder}/bpe_v${V}k.yaml"
    name="bpe_v${V}k"
    echo "==> Training ${name}"
    uv run python scripts/train_tokenizer.py --config "$baseline"

    # SuperBPE configs
    for cfg in "${folder}"/superbpe_*.yaml; do
        name="$(basename "$cfg" .yaml)"
        echo "==> Training ${name}"
        uv run python scripts/train_tokenizer.py --config "$cfg"
    done
done

echo "All tokenizers trained"

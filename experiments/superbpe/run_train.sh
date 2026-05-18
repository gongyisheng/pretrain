#!/usr/bin/env bash
# Train the SuperBPE grid sweep: 114 configs (4 BPE baselines + 110 SuperBPE).
# Smallest V first, BPE baseline before SuperBPE configs in each folder.
# Each run streams to logs/superbpe/train/<name>.log.
#
# Usage: nohup bash experiments/superbpe/run_train.sh > logs/superbpe_train.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/superbpe/train

for V in 50 100 150 200; do
    folder="experiments/superbpe/v${V}k"
    if [[ ! -d "$folder" ]]; then
        echo "error: $folder missing; run generate_configs.py first" >&2
        exit 1
    fi

    # BPE baseline first (fast sanity-check signal per V).
    baseline="${folder}/bpe_v${V}k.yaml"
    name="bpe_v${V}k"
    echo "==> Training ${name}"
    uv run python scripts/train_tokenizer.py --config "$baseline" \
        2>&1 | tee "logs/superbpe/train/${name}.log"

    # SuperBPE configs in alphabetical shell-glob order (t100k_m2 runs before
    # t20k_m2 for V ≥ 150k). Replace the glob with `sort -V` here for strict
    # t-then-m order.
    for cfg in "${folder}"/superbpe_*.yaml; do
        name="$(basename "$cfg" .yaml)"
        echo "==> Training ${name}"
        uv run python scripts/train_tokenizer.py --config "$cfg" \
            2>&1 | tee "logs/superbpe/train/${name}.log"
    done
done

echo "All 114 tokenizers trained. Per-run logs in logs/superbpe/train/."

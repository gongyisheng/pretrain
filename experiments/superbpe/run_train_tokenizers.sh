#!/usr/bin/env bash
# Train the BPE baseline + every SuperBPE transition point sequentially.
# Each run streams to logs/<name>_train.log.
# Usage: nohup bash experiments/superbpe/run_train_tokenizers.sh > logs/superbpe.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

names=(bpe_200k)
for t in 20 40 60 80 100 120 140 160 180; do
    names+=("superbpe_200k_t${t}k")
done

for name in "${names[@]}"; do
    echo "==> Training $name"
    uv run python scripts/train_tokenizer.py --config "experiments/superbpe/${name}.yaml" 
done

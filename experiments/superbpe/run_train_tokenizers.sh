#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
for cfg in experiments/superbpe/*.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "==> Training $name"
    uv run python scripts/train_tokenizer.py --config "$cfg" \
        2>&1 | tee "logs/${name}_train.log"
done

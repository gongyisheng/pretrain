#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for method in fixed lecun xavier kaiming; do
    for gain in 0.25 0.5 1 2 4; do
        stem="qwen3_51m_adamw_${method}_gain_${gain}"
        uv run python scripts/train.py --config "experiments/adamw_init_method/${stem}.yaml" "$@"
    done
done

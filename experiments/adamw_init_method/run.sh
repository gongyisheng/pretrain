#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for init_std in 0.001 0.002 0.005 0.01 0.02 0.05; do
    stem="qwen3_51m_adamw_fixed_std_${init_std}"
    uv run python scripts/train.py --config "experiments/adamw_init_method/${stem}.yaml" "$@"
done

for method in lecun xavier kaiming; do
    stem="qwen3_51m_adamw_${method}"
    uv run python scripts/train.py --config "experiments/adamw_init_method/${stem}.yaml" "$@"
done

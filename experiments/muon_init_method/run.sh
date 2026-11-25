#!/bin/bash
# Run the Muon initialization method sweep.
# Usage: nohup bash experiments/muon_init_method/run.sh > logs/muon_init_method.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

configs=()
for method in fixed lecun xavier kaiming; do
    if [[ "$method" == fixed ]]; then
        init_stds=(0.001 0.002 0.005 0.01 0.02 0.05)
    else
        init_stds=(0.02)
    fi
    for init_std in "${init_stds[@]}"; do
        stem="qwen3_51m_muon_${method}"
        if [[ "$method" == fixed ]]; then
            stem+="_std_${init_std}"
        fi
        configs+=("experiments/muon_init_method/${stem}.yaml")
    done
done

for config in "${configs[@]}"; do
    uv run python scripts/train.py --config "$config" "$@"
done

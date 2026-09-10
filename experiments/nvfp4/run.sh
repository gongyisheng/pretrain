#!/bin/bash
# Run the Qwen3-51M BF16 baseline and NVFP4 recipe comparison.
# Usage: nohup bash experiments/nvfp4/run.sh > logs/nvfp4_51m.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
precisions=(bf16 nvfp4)

configs=()
for model in "${models[@]}"; do
    for precision in "${precisions[@]}"; do
        configs+=("${model}_${precision}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/nvfp4/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== NVFP4 runs complete ==="

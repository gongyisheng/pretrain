#!/bin/bash
# Run FP8 (3 recipes) vs bf16 experiment at 0.5B on Qwen3.
# Usage: nohup bash experiments/fp8/run_0.5b.sh > logs/fp8_0.5b.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_0.5b_bf16)
for recipe in tensorwise rowwise rowwise_with_gw_hp; do
    configs+=("qwen3_0.5b_fp8_${recipe}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 0.5B FP8 runs complete ==="

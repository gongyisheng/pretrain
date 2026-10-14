#!/bin/bash
# Compare W4A16 while keeping selected Transformer blocks in bf16.
# Usage: nohup bash experiments/int4_weight_selective_layers/run.sh > logs/int4_weight_selective_layers.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
skip_variants=(no_skip skip_7 skip_6_7 skip_0_7 skip_0_6_7)
configs=()

for model in "${models[@]}"; do
    configs+=("${model}_bf16")
    for skip_variant in "${skip_variants[@]}"; do
        configs+=("${model}_int4_w4a16_${skip_variant}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_selective_layers/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== int4_weight_selective_layers runs complete ==="

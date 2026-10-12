#!/bin/bash
# Compare W4A16 while keeping selected Transformer blocks in bf16.

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
variants=(bf16 int4_w4a16_no_skip int4_w4a16_skip_7 int4_w4a16_skip_6_7 int4_w4a16_skip_0_7)
configs=()

for model in "${models[@]}"; do
    for variant in "${variants[@]}"; do
        configs+=("${model}_${variant}")
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

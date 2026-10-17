#!/bin/bash
# Compare bf16 with W4A16 quantization disabled for the final updates.
# Usage: nohup bash experiments/int4_weight_high_precision_end/run.sh > logs/int4_weight_high_precision_end.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

models=(qwen3_51m)
skip_variants=(no_skip skip_0_6_7)
end_steps=(49500 49000 47500 45000 40000)
configs=()

for model in "${models[@]}"; do
    configs+=("${model}_bf16")
    for skip_variant in "${skip_variants[@]}"; do
        for end_step in "${end_steps[@]}"; do
            if [[ "${skip_variant}" == "no_skip" ]]; then
                configs+=("${model}_int4_w4a16_end_${end_step}")
            else
                configs+=("${model}_int4_w4a16_${skip_variant}_end_${end_step}")
            fi
        done
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_high_precision_end/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== int4_weight_high_precision_end runs complete ==="

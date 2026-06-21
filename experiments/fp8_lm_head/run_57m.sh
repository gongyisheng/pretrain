#!/bin/bash
# Run FP8 (3 recipes, lm_head INCLUDED in FP8) vs bf16 at 57M on Qwen3.
# Usage: nohup bash experiments/fp8_lm_head/run_57m.sh > logs/fp8_lm_head_57m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_57m_bf16)
for recipe in tensorwise rowwise rowwise_with_gw_hp; do
    configs+=("qwen3_57m_fp8_${recipe}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_lm_head/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 57M FP8 lm_head runs complete ==="

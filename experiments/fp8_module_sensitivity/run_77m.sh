#!/bin/bash
# FP8 module-sensitivity add-one-in ablation at Qwen3-77M (untied), Muon optimizer.
# Shared bf16 baseline, then each module (attn/mlp/lm_head) in FP8 under three
# recipes: w8a16 (weight-only), w8a8 (fwd fp8, bf16 backward), w8a8g8 (full fp8).
# Usage: nohup bash experiments/fp8_module_sensitivity/run_77m.sh > logs/fp8_module_sensitivity_77m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

DIR=experiments/fp8_module_sensitivity
recipes=(w8a16 w8a8 w8a8g8)
modules=(attn mlp lm_head)

configs=("${DIR}/qwen3_77m_bf16.yaml")
for recipe in "${recipes[@]}"; do
    for module in "${modules[@]}"; do
        configs+=("${DIR}/${recipe}/qwen3_77m_${recipe}_${module}.yaml")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 77M fp8_module_sensitivity runs complete ==="

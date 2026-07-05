#!/bin/bash
# FP8 module-sensitivity ablation at Qwen3-77M (untied), Muon optimizer.
# Add-one-in: bf16 baseline, then each module group quantized alone, then all.
# Usage: nohup bash experiments/fp8_module_sensitivity/run.sh > logs/fp8_module_sensitivity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 fp8_attn fp8_mlp fp8_lm_head fp8_all)
configs=()
for v in "${variants[@]}"; do
    configs+=("qwen3_77m_${v}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_module_sensitivity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 77M fp8_module_sensitivity runs complete ==="

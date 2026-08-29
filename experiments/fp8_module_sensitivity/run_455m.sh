#!/bin/bash
# FP8 module-sensitivity add-one-in ablation at Qwen3-455M (untied), Muon optimizer.
# Runs on H200 (144GiB): batch_size=32, grad_accum=8 (eff. batch 256).
# Usage: nohup bash experiments/fp8_module_sensitivity/run_455m.sh > logs/fp8_module_sensitivity_455m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 fp8_attn fp8_mlp fp8_lm_head)
configs=()
for v in "${variants[@]}"; do
    configs+=("qwen3_455m_${v}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_module_sensitivity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 455M fp8_module_sensitivity runs complete ==="

#!/bin/bash
# Int8 module-sensitivity add-one-in ablation at Qwen3-455M (untied), Muon optimizer.
# Runs on H200 (144GiB): batch_size=32, grad_accum=8 (eff. batch 256).
# Usage: nohup bash experiments/int8_module_sensitivity/run_455m.sh > logs/int8_module_sensitivity_455m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 w8a16_attn w8a16_mlp w8a16_lm_head)
configs=()
for v in "${variants[@]}"; do
    configs+=("qwen3_455m_${v}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_module_sensitivity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 455M int8_module_sensitivity runs complete ==="

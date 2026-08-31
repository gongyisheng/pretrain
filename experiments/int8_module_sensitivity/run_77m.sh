#!/bin/bash
# Int8 module-sensitivity add-one-in ablation at Qwen3-77M (untied), Muon optimizer.
# Usage: nohup bash experiments/int8_module_sensitivity/run_77m.sh > logs/int8_module_sensitivity_77m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 w8a16_attn w8a16_mlp w8a16_lm_head)
configs=()
for v in "${variants[@]}"; do
    configs+=("qwen3_77m_${v}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_module_sensitivity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 77M int8_module_sensitivity runs complete ==="

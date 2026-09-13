#!/bin/bash
# Qwen3-51M bf16 baseline plus int4 W4A16 with RNE and with stochastic weight rounding, both at (1, 16) blockwise 1D.
# Usage: nohup bash experiments/int4_stochastic_rounding/run.sh > logs/int4_stochastic_rounding_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

rounding=(rne sr)

configs=(qwen3_51m_bf16)
for mode in "${rounding[@]}"; do
    configs+=("qwen3_51m_int4_w4a16_${mode}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_stochastic_rounding/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_stochastic_rounding runs complete ==="

#!/bin/bash
# Qwen3-51M bf16 baseline plus int4 W4A16 with fp32 block scales and with e4m3 block scales + fp32 global scale.
# Usage: nohup bash experiments/int4_global_scale/run.sh > logs/int4_global_scale_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

arms=(bs_fp32 bs_fp8_e4m3_gs_fp32)

configs=(qwen3_51m_bf16)
for arm in "${arms[@]}"; do
    configs+=("qwen3_51m_int4_w4a16_${arm}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_global_scale/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_global_scale runs complete ==="

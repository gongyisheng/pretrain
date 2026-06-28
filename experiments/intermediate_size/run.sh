#!/bin/bash
# Sweep FFN expansion ratio (intermediate_size / d_model) with Muon at fixed lr=5e-4.
# 15 widths (mult 0.25 - 32, dense 128-step grid over 1024-2048)
# Usage: nohup bash experiments/intermediate_size/run.sh > logs/intermediate_size.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

# Config filenames embed total param count, which grows with intermediate_size.
configs=(
    qwen3_34m_is128
    qwen3_35m_is256
    qwen3_38m_is512
    qwen3_45m_is1024
    qwen3_46m_is1152
    qwen3_48m_is1280
    qwen3_49m_is1408
    qwen3_51m_is1536
    qwen3_53m_is1664
    qwen3_54m_is1792
    qwen3_56m_is1920
    qwen3_57m_is2048
    qwen3_82m_is4096
    qwen3_133m_is8192
    qwen3_233m_is16384
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/intermediate_size/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All intermediate-size sweep runs complete ==="

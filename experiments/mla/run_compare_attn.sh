#!/bin/bash
# Phase 1 — attention comparison: MLA (center config) vs GQA vs MHA on Qwen3 57M, QK score dim matched across all three.
# Usage: nohup bash experiments/mla/run_compare_attn.sh > logs/mla_compare_attn.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_57m_mha
    qwen3_57m_gqa
    qwen3_57m_mla
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 1 (compare attn) complete ==="

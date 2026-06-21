#!/bin/bash
# Phase 1 — attention comparison: MLA vs GQA vs MHA on Qwen3 57M.
# MLA uses the center config (nope=32, rope=32, v_head=64, kv_lora=512); its
# per-head QK score dim (32+32=64) matches the MHA/GQA head_dim for a clean
# apples-to-apples comparison. Each variant is its own standalone YAML.
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

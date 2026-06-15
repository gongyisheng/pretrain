#!/bin/bash
# Phase 3 — qk_nope:qk_rope split at fixed total head dim = 64 (v_head=64,
# kv_lora=512 fixed). Tests how to divide the 64-dim QK budget between content
# matching (nope) and decoupled position (rope). 32:32 = center (run in Phase 1),
# so it is not repeated here.
# Usage: nohup bash experiments/mla/run_qk_nope_rope.sh > logs/mla_qk_nope_rope.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for split in 8_rope_56 16_rope_48 24_rope_40 40_rope_24 48_rope_16 56_rope_8; do
    configs+=("qwen3_57m_mla_nope_${split}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 3 (qk_nope:qk_rope) complete ==="

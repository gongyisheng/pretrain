#!/bin/bash
# Phase 4 — v_head_dim sweep (nope=32, rope=32, kv_lora=512 fixed). Varies the
# value/output head width. Note kv_lora=512 is held fixed, so larger v_head both
# widens values AND raises the latent's reconstruction target 8*(32+v_head) —
# at v_head=256 the latent compresses 4.5x, so this axis is not purely "value
# capacity". 64 = center (run in Phase 1), so it is not repeated here.
# Usage: nohup bash experiments/mla/run_v_head_dim.sh > logs/mla_v_head_dim.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for v in 16 32 128 256; do
    configs+=("qwen3_57m_mla_vhead_${v}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 4 (v_head_dim) complete ==="

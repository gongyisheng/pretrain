#!/bin/bash
# Phase 5 — q_lora_rank sweep (nope=32, rope=32, v_head=64, kv_lora=512 fixed).
# Query compression: q is factored through a q_lora_rank latent instead of a
# direct projection. The q output is n_heads*qk_head=8*64=512, so r=512 is
# rank-equivalent to the direct q_proj (modulo q_a_norm) and r<256 saves params.
# This is a params/compute knob, not a KV-cache knob. center (q_lora=0, direct
# projection) is the reference and is run in Phase 1, so it is not repeated here.
# Usage: nohup bash experiments/mla/run_q_lora_rank.sh > logs/mla_q_lora_rank.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for q in 64 128 256 384 512; do
    configs+=("qwen3_57m_mla_qlora_${q}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 5 (q_lora_rank) complete ==="

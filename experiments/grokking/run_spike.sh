#!/usr/bin/env bash
# Slingshot-spike ablation for grokking (sub task).
#
# Tests what eliminates the slingshot spikes observed in the wd>0 regime.
# Two complementary hypotheses:
#
#   A) The spike is driven by fp32 CE precision collapse (Liu et al. 2025,
#      arXiv:2605.06152). Use cross_entropy_fp64 to restore gradient direction.
#   B) The spike is driven by AdamW's 1/sqrt(v) amplification when v collapses
#      between memorization and wd-driven basin exit. Switch to Lion (no v,
#      bounded sign-momentum step) to remove the amplification path.
#
# Variants:
#   wd0.0_ce            - fp32 CE,  AdamW, wd=0   (paper's regime;   expected: spikes)
#   wd0.0_ce_fp64       - fp64 CE,  AdamW, wd=0   (paper's fix;      expected: no spikes)
#   wd0.1_ce_fp64       - fp64 CE,  AdamW, wd=0.1 (our regime;       observed: spikes)
#   wd0.1_ce_fp64_lion  - fp64 CE,  Lion,  wd=0.1 (matched-wd to spike regime)
#   wd0.3_ce_fp64_lion  - fp64 CE,  Lion,  wd=0.3 (matched effective decay: lr*wd ≈ AdamW baseline)
#
# Configs live in experiments/grokking/spike/.
#
# Tokenizer and sub-task data are prepared on demand via prepare.sh (idempotent).
#
# Usage:
#   # all 3 variants in parallel
#   nohup bash experiments/grokking/run_spike.sh > logs/grokking_spike.log 2>&1 &
#
#   # single variant
#   nohup bash experiments/grokking/run_spike.sh wd0.0_ce_fp64 > logs/grokking_spike.log 2>&1 &
#
#   # pin to a GPU
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run_spike.sh > logs/grokking_spike.log 2>&1 &
#
# MAX_CONCURRENCY (default 3) controls parallelism. Each run gets its own
# per-config log under logs/grokking/spike_<variant>.log.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(wd0.0_ce wd0.0_ce_fp64 wd0.1_ce_fp64 wd0.1_ce_fp64_lion wd0.3_ce_fp64_lion wd0.1_ce_fp64_ls1e-5 wd0.1_ce_ls1e-5)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-5}"
PER_RUN_LOG_DIR="logs/grokking"

bash experiments/grokking/prepare.sh sub

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/spike/qwen3_1m_sub_${variant}.yaml"
    local log="${PER_RUN_LOG_DIR}/spike_${variant}.log"
    echo "[run_spike.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 1 ]; then
    run_one "$1"
    exit 0
fi

echo "[run_spike.sh] launching spike ablation with MAX_CONCURRENCY=$MAX_CONCURRENCY"
running=0
for variant in "${VARIANTS[@]}"; do
    run_one "$variant" &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_CONCURRENCY" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

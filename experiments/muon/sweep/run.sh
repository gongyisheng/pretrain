#!/bin/bash
# Run all Muon param sweep experiments on 16M sequentially.
# The default (lr=0.04, momentum=0.95, backend_steps=5, embed_lr=0.6) is
# covered by experiments/muon/gpt2_16m_muon.yaml — run that first as baseline.
# Usage: nohup bash experiments/muon/sweep/run.sh > sweep.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

echo "=== lr sweep ==="
for config in lr_0.01 lr_0.02 lr_0.08; do
    echo "--- ${config} --- $(date)"
    python scripts/train.py --config "experiments/muon/sweep/${config}.yaml"
done

echo "=== momentum sweep ==="
for config in momentum_0.85 momentum_0.90 momentum_0.98; do
    echo "--- ${config} --- $(date)"
    python scripts/train.py --config "experiments/muon/sweep/${config}.yaml"
done

echo "=== backend_steps sweep ==="
for config in backend_steps_3 backend_steps_10; do
    echo "--- ${config} --- $(date)"
    python scripts/train.py --config "experiments/muon/sweep/${config}.yaml"
done

echo "=== embed_lr sweep ==="
for config in embed_lr_0.3 embed_lr_1.0; do
    echo "--- ${config} --- $(date)"
    python scripts/train.py --config "experiments/muon/sweep/${config}.yaml"
done

echo "=== All sweep runs complete ==="

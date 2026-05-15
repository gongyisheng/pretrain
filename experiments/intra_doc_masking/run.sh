#!/bin/bash
# Run all intra_doc_masking seqlen experiments sequentially
# Usage: nohup bash experiments/intra_doc_masking/run.sh > logs/intra_doc_masking.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

SEQ_LENS=(2048 4096 8192)
VARIANTS=(baseline masked)

for seqlen in "${SEQ_LENS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        config="experiments/intra_doc_masking/qwen3_57m_seqlen_${seqlen}_${variant}.yaml"
        echo "=== seqlen=${seqlen} ${variant} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "${config}"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== All intra_doc_masking runs complete ==="

#!/usr/bin/env bash

echo "=== Run 1: baseline (no doc_mask) ==="
uv run python scripts/train.py --config experiments/intra_doc_masking/qwen3_57m_baseline.yaml

echo "=== Run 2: doc_mask enabled ==="
uv run python scripts/train.py --config experiments/intra_doc_masking/qwen3_57m_masked.yaml

echo "=== Done. Compare runs in W&B under group: cross-doc-sweep ==="

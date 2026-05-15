# Intra-Doc Masking

Measure the effect of intra-document attention masking when packed sequences contain multiple documents, and how the effect scales with sequence length.

## Hypothesis

With document packing, a single sequence concatenates many documents separated by `eot` tokens. Default causal attention lets each token attend to every earlier token in the pack, including tokens from unrelated prior documents. Intra-doc masking restricts attention to within the current document only.

At short seq_len, packs contain few documents — cross-doc attention is rare and the effect should be small. At long seq_len, packs hold many documents — masking should matter more, either as a quality win (cleaner signal) or a quality loss (less context for in-context patterns to emerge).

## Setup

Fixed: Qwen3 57M architecture, `batch_size=16`, `gradient_accumulation_steps=16`, `lr=6e-4`, 50K steps, `bf16`.

Each `seq_len` is run twice — once with `intra_doc_masking: false` (baseline), once with `true` (masked).

| Config | seq_len | intra_doc_masking | Tokens/step |
|---|---|---|---|
| qwen3_57m_seqlen_2048_baseline | 2048 | false | ~524K |
| qwen3_57m_seqlen_2048_masked   | 2048 | true  | ~524K |
| qwen3_57m_seqlen_4096_baseline | 4096 | false | ~1.05M |
| qwen3_57m_seqlen_4096_masked   | 4096 | true  | ~1.05M |
| qwen3_57m_seqlen_8192_baseline | 8192 | false | ~2.10M |
| qwen3_57m_seqlen_8192_masked   | 8192 | true  | ~2.10M |

## Run

```bash
nohup bash experiments/intra_doc_masking/run.sh > logs/intra_doc_masking.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/intra_doc_masking/qwen3_57m_seqlen_4096_masked.yaml
```

## W&B

Project: `pretrain-intra-doc-masking`. Compare baseline vs masked at each seq_len by `val/loss` vs `train/step`.

## Results

TODO

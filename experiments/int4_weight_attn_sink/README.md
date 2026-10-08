# Int4 Weight Attention-Sink Ablation

## Hypothesis

Attention sinks may reduce the validation-loss penalty from W4A16 training by letting each query head assign probability mass to a learned zero-value destination. This four-way ablation measures the sink effect within BF16 and W4A16, and whether sinks reduce the quantization penalty.

## Setup

| Config | Weights / activations | Attention sink | Approx. parameters | Key settings |
|---|---|---:|---:|---|
| [qwen3_51m_bf16_attn_sink_off.yaml](qwen3_51m_bf16_attn_sink_off.yaml) | bf16 / bf16 | false | 50,931,200 | FlexAttention |
| [qwen3_51m_bf16_attn_sink_on.yaml](qwen3_51m_bf16_attn_sink_on.yaml) | bf16 / bf16 | true | 50,931,264 | FlexAttention |
| [qwen3_51m_w4a16_attn_sink_off.yaml](qwen3_51m_w4a16_attn_sink_off.yaml) | int4 / bf16 | false | 50,931,200 | FlexAttention, W4A16 |
| [qwen3_51m_w4a16_attn_sink_on.yaml](qwen3_51m_w4a16_attn_sink_on.yaml) | int4 / bf16 | true | 50,931,264 | FlexAttention, W4A16 |

All runs use Qwen3-51M (8 layers, `d_model=512`, GQA 8/4 heads, QK norm, MLP size 1536), OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split, sequence length 1024, batch size 16, gradient accumulation 16 (262,144 tokens/step), 50,000 steps, bf16 mixed precision, and seed 42. MuonAdam uses lr=5e-4 and weight decay=0.1, with cosine decay, 1,500 warmup steps, and min lr=5e-5. Checkpoints are saved every 5,000 steps; evaluation runs every 100 steps for 100 batches.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/int4_weight_attn_sink/run.sh
```

Additional trainer arguments are forwarded to every run.

## Results

| Config | Mean validation loss, final 10 evals | Tokens/s | Notes |
|---|---:|---:|---|
| bf16, sink off | Pending | Pending | BF16 reference |
| bf16, sink on | Pending | Pending | Sink delta vs. bf16 sink off: pending |
| W4A16, sink off | Pending | Pending | Quantization penalty vs. bf16 sink off: pending |
| W4A16, sink on | Pending | Pending | Quantization penalty vs. bf16 sink on; sink delta vs. W4A16 sink off: pending |

Training has not been run. Report sink delta as loss(on) minus loss(off) within each precision, and quantization penalty as loss(W4A16) minus loss(BF16) at each sink setting. Negative sink deltas indicate improvement; a smaller quantization penalty with sinks on supports the hypothesis. Compare tokens/s over the same post-warmup steps.

## Notes

This is a single-seed comparison. Every configuration explicitly uses FlexAttention; enabling sinks adds 64 zero-initialized learned logits (8 layers × 8 query heads). W4A16 uses RNE int4 weights with bf16 activations and output gradients, blockwise 16×16 fp8_e4m3 scales with a global scale, and randomized-sign Hadamard rotation with block size 16 on both weight axes in forward and dgrad. Activation and output-gradient rotation axes are empty. Quantization applies to every eligible attention and MLP linear; only `lm_head` is excluded.

The runner executes BF16 on/off, then W4A16 on/off. W&B project: `pretrain-int4-weight-attn-sink`; checkpoints: `checkpoints/int4_weight_attn_sink/<config>/`.

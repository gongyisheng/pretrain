# INT4 Activation Granularity

Compare rowwise, 1D, and square 2D activation granularities during INT4 pretraining, holding weight quantization fixed. Unlike the joint weight/activation sweeps, only `training.quantization.scale.act` changes across the W4A4 configs.

## Hypothesis

With a 32-element contraction extent, a 1D block shares a scale over 32 values while a 2D block shares it over 1,024 values. We expect 1D blocks to reduce activation quantization error and validation loss by containing outliers more locally. Rowwise activations use one scale across a projection's full channel dimension, which is coarser along channels than 1D blocks but does not couple tokens; there is no strict expected ordering between rowwise and 2D blocks.

## Setup

Five runs: three W4A4 activation granularities, one BF16 baseline, and one W4A16 weight-only control. `w4` and `a4` denote INT4 quantization; the codes are stored in `torch.int8`. All quantized runs use fixed 32×32 blockwise INT4 weights, FP32 scales, round-to-nearest-even, unquantized BF16 `grad_out`, and exclude `lm_head`. No activation clipping or Hadamard rotation is enabled.

| Config (`.yaml`) | Weight | Activation block | Forward values per activation scale |
|---|---|---|---|
| `qwen3_51m_bf16` | BF16 | BF16 | — |
| `qwen3_51m_int4_w4a16` | INT4 (32, 32) | BF16 | — |
| `qwen3_51m_int4_w4a4_act_rowwise` | INT4 (32, 32) | rowwise | 512 / 1,536 (projection-dependent) |
| `qwen3_51m_int4_w4a4_act_blockwise1d_32` | INT4 (32, 32) | (1, 32) | 32 |
| `qwen3_51m_int4_w4a4_act_blockwise2d_32` | INT4 (32, 32) | (32, 32) | 1,024 |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer, 50,931,200 parameters (approximately 51M) |
| Dimensions | `d_model=512`, 8 layers, 8 Q / 4 KV heads, QK norm, SwiGLU intermediate size 1,536 |
| Data | OpenWebText, `tokenizers/custom_bpe_50k`, validation split 0.01 |
| Sequence / batch | 1,024 tokens, batch 128, accumulation 2; 262,144 tokens per optimizer step |
| Budget | 50,000 steps; 13.1072B training tokens per run |
| Optimizer | Muon, momentum 0.95, Nesterov, `match_rms_adamw`, weight decay 0.1 |
| Learning rate | 5e-4, cosine decay to 5e-5, 1,500 warmup steps |
| Precision / clipping | BF16 mixed precision, gradient norm clip 1.0 |
| Seed | 42 for initialization and data ordering |
| Evaluation | Every 100 steps, 100 batches, evaluation batch size 16 |
| Checkpoints / logging | Every 5,000 / 10 steps; quantization metrics enabled |

The batch size and accumulation intentionally match the current INT8 activation-granularity experiment.

## Run

From the repository root, inspect GPU usage and select a free device. Prepare the tokenizer and OpenWebText data as described in the repository README.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free GPU from the output above.
mkdir -p logs
nohup bash experiments/int4_activation_granularity/run.sh > logs/int4_activation_granularity.log 2>&1 &
```

The launcher runs all five configurations sequentially on the selected device. W&B project: `pretrain-int4-activation-granularity`. Checkpoints: `checkpoints/int4_activation_granularity/<config>/`.

## Results

Primary metric: mean `val/loss` over the final ten scheduled evaluations, at steps 49,100–50,000. Compare runs at equal training tokens. Record validation BPB, per-module activation SQNR and underflow, and any nonfinite loss or divergence step. Current activation statistics pool forward and weight-gradient observations. Results are pending.

| Config suffix | Mean val loss | Δ vs BF16 | Δ vs W4A16 | Val BPB | Status |
|---|---|---|---|---|---|
| `bf16` | — | 0 | — | — | Pending |
| `int4_w4a16` | — | — | 0 | — | Pending |
| `int4_w4a4_act_rowwise` | — | — | — | — | Pending |
| `int4_w4a4_act_blockwise1d_32` | — | — | — | — | Pending |
| `int4_w4a4_act_blockwise2d_32` | — | — | — | — | Pending |

Report `loss(2D, 32) - loss(1D, 32)`; positive values favor 1D. Compare each W4A4 run with W4A16 to measure the added activation-quantization cost under the fixed weight policy. These trained-model differences include changes in optimization trajectories.

## Notes

- Validation bypasses quantization in `QuantizedLinear.eval()`. Validation loss therefore measures the effect of quantized training on the learned model, evaluated in BF16; it does not measure quantized inference loss.
- **Arithmetic paths:** INT4 codes are stored as INT8. The blockwise W4A4 runs have matching 32-value contraction extents and use the scaled INT8 GEMM path on those codes. Rowwise W4A4 activations have a rowwise extent while weights use 32, so they quantize/dequantize and use BF16 matmul. W4A16 also uses BF16 matmul because its activations are unquantized. The quantized linear layers use BF16 backward GEMMs after quantizing and dequantizing the INT4 operands; `grad_out` stays unquantized. Do not use this experiment for pure timing or scale-only numerical comparisons across runs.
- Fixed 32×32 weight blocks give both blockwise W4A4 runs the same 32-value contraction extent. Learned weights and their quantization errors can still diverge across runs.
- In forward, activations are flattened to `(batch × sequence, channels)`: 1D groups 32 channels of one token; 2D groups 32 tokens × 32 channels. In weight-gradient computation, the contraction axis is tokens, so 1D groups 32 tokens of one channel. The sweep affects activation quantization in both forward and weight-gradient computation; it does not isolate forward-only sensitivity.
- Rowwise activation quantization shares a forward scale across a projection's full input channel dimension (512 or 1,536 values) and a weight-gradient scale across all flattened tokens. The 2D block shares each scale across 32 times more values than the 1D block. This compares practical granularity choices, not geometry at equal scale count. Logical FP32 scale overhead is 1 bit per activation for 1D and 0.03125 bits per activation for 2D; this is not measured memory usage, because the implementation can expand scales and retains other training tensors.
- Seed 42 is a screening sweep. Before claiming a small difference, repeat all three W4A4 configurations and both controls with seeds 43 and 44, using `--training.seed`, `--training.checkpoint_dir`, and `--logging.wandb_run_name` overrides with distinct paths and names. Report paired loss differences and spread across seeds; ten evaluations within one run are not independent replicates.
- Do not reuse older experiment baselines with different microbatch sizes or quantization policies. Check each run reaches 50,000 steps: `scripts/train.py` currently catches training exceptions without returning a failing exit status, so launcher completion alone does not prove successful training.

# FP8 Scaling-Granularity Sweep

Sweep FP8 scale granularity at Qwen3-51M against a bf16 baseline. The primary comparison holds the FP8 dtype recipe fixed and varies only scale layout.

## Hypothesis

Smaller FP8 scale blocks reduce quantization error and improve validation loss, at the cost of scale storage and kernel overhead. `tensorwise` is the cheapest and most aggressive setting; rowwise and blockwise settings should trade throughput for accuracy.

## Setup

| Config | Granularity | `block_shape` | W/A/G format | Approx params |
|---|---|---|---|---|
| `qwen3_51m_bf16` | bf16 | — | bf16 | ~51M |
| `qwen3_51m_fp8_tensorwise` | tensorwise | — | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_rowwise` | rowwise | — | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise1d_16` | blockwise 1D | `(1, 16)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise1d_32` | blockwise 1D | `(1, 32)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise1d_64` | blockwise 1D | `(1, 64)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise1d_128` | blockwise 1D | `(1, 128)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise2d_16` | blockwise 2D | `(16, 16)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise2d_32` | blockwise 2D | `(32, 32)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise2d_64` | blockwise 2D | `(64, 64)` | E4M3/E4M3/E5M2 | ~51M |
| `qwen3_51m_fp8_blockwise2d_128` | blockwise 2D | `(128, 128)` | E4M3/E4M3/E5M2 | ~51M |

All configs use ~51M parameters: `d_model=512`, 8 layers, 8/4 Q/KV heads, `intermediate_size=1536`, and `max_seq_len=1024`. They use OpenWebText, batch size 16, gradient accumulation 16, 50K steps, bf16 mixed precision, Muon (`match_rms_adamw`, momentum 0.95, Nesterov), learning rate 5e-4, cosine decay with 1,500 warmup steps, and minimum learning rate 5e-5. Checkpoints are every 5,000 steps; evaluation is every 100 steps for 25 batches.

The primary FP8 configs explicitly set E4M3 for weights and activations and E5M2 for output gradients. `lm_head` remains bf16 in every quantized run. Tensorwise, rowwise, and blockwise FP8 GEMMs dispatch to the in-house Triton scaled-GEMM backend on SM 8.9+ GPUs; they do not use the cuBLASLt MXFP8 path.

## Run

The primary sweep runs 11 recipes; the weight-transpose ablation adds four.

```bash
nohup bash experiments/fp8_granularity/run.sh > logs/fp8_granularity_51m.log 2>&1 &
```

## Results

| Model | Precision | Granularity | `block_shape` | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | — | | | | 1.00× |
| 51M | fp8 | tensorwise | — | | | | |
| 51M | fp8 | rowwise | — | | | | |
| 51M | fp8 | blockwise 1D | (1, 16) | | | | |
| 51M | fp8 | blockwise 1D | (1, 32) | | | | |
| 51M | fp8 | blockwise 1D | (1, 64) | | | | |
| 51M | fp8 | blockwise 1D | (1, 128) | | | | |
| 51M | fp8 | blockwise 2D | (16, 16) | | | | |
| 51M | fp8 | blockwise 2D | (32, 32) | | | | |
| 51M | fp8 | blockwise 2D | (64, 64) | | | | |
| 51M | fp8 | blockwise 2D | (128, 128) | | | | |

## Notes

- Compare FP8 runs with the bf16 baseline using the mean validation loss over the final 10 evaluations.
- At this model size, non-GEMM kernels can limit end-to-end throughput gains.

## Weight-transpose consistency

This separate ablation holds the FP8 block extent at 32 and keeps activations in bf16. It compares 1D blocks, which re-partition a weight when it is transposed between forward and dgrad, with square 2D blocks, whose tile partition transposes consistently.

### Hypothesis

If inconsistent weight blocking is a material error source, `(32, 32)` should improve loss or stability over `(1, 32)`. The G16 pair isolates this effect with bf16 output gradients; the G8 pair tests whether quantizing the output-gradient operand changes that conclusion.

### Setup

| Config | Weight / activation / grad-out | `block_shape` | GEMM behavior | Approx params |
|---|---|---|---|---|
| `qwen3_51m_fp8_blockwise1d_32_w8a16g16` | E4M3 / bf16 / bf16 | `(1, 32)` | fake-quant + bf16 GEMMs | ~51M |
| `qwen3_51m_fp8_blockwise2d_32_w8a16g16` | E4M3 / bf16 / bf16 | `(32, 32)` | fake-quant + bf16 GEMMs | ~51M |
| `qwen3_51m_fp8_blockwise1d_32_w8a16g8` | E4M3 / bf16 / E4M3 | `(1, 32)` | two quantized operands only in dgrad | ~51M |
| `qwen3_51m_fp8_blockwise2d_32_w8a16g8` | E4M3 / bf16 / E4M3 | `(32, 32)` | two quantized operands only in dgrad | ~51M |

All four use fp32 scales and exclude `lm_head`; their batch size, gradient accumulation, checkpoint, and evaluation settings match the primary sweep. G16 has at most one quantized operand per GEMM, so quantization is dequantized before each bf16 matmul. In G8, forward and wgrad still have one quantized operand; dgrad is the only GEMM with two quantized operands.

### Run

`run.sh` runs this ablation after the primary sweep.

### Results

| Model | Grad-out | Granularity | `block_shape` | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|---|
| 51M | bf16 | blockwise 1D | `(1, 32)` | | | | |
| 51M | bf16 | blockwise 2D | `(32, 32)` | | | | |
| 51M | E4M3 | blockwise 1D | `(1, 32)` | | | | |
| 51M | E4M3 | blockwise 2D | `(32, 32)` | | | | |

### Notes

- Compare 1D and 2D within each grad-out precision before comparing G16 with G8.
- Record instability, divergence, or loss spikes in addition to final-window validation loss.

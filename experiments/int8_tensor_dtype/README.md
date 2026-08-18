# Integer Tensor-Dtype Sensitivity

Compare Qwen3-51M pretraining in bf16 with rowwise integer quantization from int8 through int4. At each effective element width, run W8A16 (weight only) and W8A8 (weight plus activation); `grad_out` remains bf16. The independent variables are element width and the selected tensors in `training.quantization.dtype`. Design: `docs/superpowers/specs/2026-08-17-quant-operand-and-int-bit-allocation-design.md`.

`W8A16` and `W8A8` name the storage layout: the integer codes use an int8 container and the unquantized tensor is bf16. The `int8` … `int4` config prefix sets the effective code range. Thus `int4_w8a8` has int4-range codes stored in int8 tensors for weight and activation.

```
fwd     y  = act       @ weightᵀ
dgrad   dx = grad_out  @ weight
wgrad   dw = grad_outᵀ @ act
```

Quantizing weight affects forward and dgrad; quantizing act affects forward and wgrad. Both cells leave `grad_out` bf16.

## Hypothesis

W8A16 should remain close to bf16 at int8 and lose accuracy progressively at lower widths. W8A8 tests whether activation quantization moves that knee upward. Its forward GEMM has two integer operands and uses the scaled INT8 GEMM path; W8A16 uses fake quantization followed by bf16 matmul.

## Setup

All hyperparameters match. Every quantized run uses `scaling: {granularity: rowwise}`, `exclude: [lm_head]`, seed 42, and the same data order and LR schedule.

| Config | weight | act | grad_out | Approx params |
|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | bf16 | bf16 | ~51M |
| qwen3_51m_int8_w8a16 | int8 | bf16 | bf16 | ~51M |
| qwen3_51m_int8_w8a8 | int8 | int8 | bf16 | ~51M |
| qwen3_51m_int7_w8a16 | int7 | bf16 | bf16 | ~51M |
| qwen3_51m_int7_w8a8 | int7 | int7 | bf16 | ~51M |
| qwen3_51m_int6_w8a16 | int6 | bf16 | bf16 | ~51M |
| qwen3_51m_int6_w8a8 | int6 | int6 | bf16 | ~51M |
| qwen3_51m_int5_w8a16 | int5 | bf16 | bf16 | ~51M |
| qwen3_51m_int5_w8a8 | int5 | int5 | bf16 | ~51M |
| qwen3_51m_int4_w8a16 | int4 | bf16 | bf16 | ~51M |
| qwen3_51m_int4_w8a8 | int4 | int4 | bf16 | ~51M |

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 50K steps (~13B tokens), Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, OpenWebText, bf16 mixed precision, and `eval_steps: 100`.

**Readout.** Compare mean validation loss over the final 10 evals with this experiment's bf16 anchor. Read W8A16 down the bit-width axis for weight sensitivity; compare W8A8 with W8A16 at a width for the added activation effect.

## Run

```bash
nohup bash experiments/int8_tensor_dtype/run.sh > logs/int8_tensor_dtype.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-tensor-dtype`.

| Model | width | cell | int tensor | GEMMs touched | Val Loss | Δ vs bf16 | Val BPB | rel. quant err |
|---|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | — | — | | 0 | | — |
| 51M | int8 | W8A16 | weight | fwd, dgrad | | | | |
| 51M | int8 | W8A8 | weight + act | fwd, dgrad, wgrad | | | | |
| 51M | int7 | W8A16 | weight | fwd, dgrad | | | | |
| 51M | int7 | W8A8 | weight + act | fwd, dgrad, wgrad | | | | |
| 51M | int6 | W8A16 | weight | fwd, dgrad | | | | |
| 51M | int6 | W8A8 | weight + act | fwd, dgrad, wgrad | | | | |
| 51M | int5 | W8A16 | weight | fwd, dgrad | | | | |
| 51M | int5 | W8A8 | weight + act | fwd, dgrad, wgrad | | | | |
| 51M | int4 | W8A16 | weight | fwd, dgrad | | | | |
| 51M | int4 | W8A8 | weight + act | fwd, dgrad, wgrad | | | | |

## Notes

- Everything outside eligible `nn.Linear` modules remains bf16/fp32; `lm_head` is excluded.
- Rowwise scaling is per-token for the forward activation and per-channel for the forward weight. In the backward GEMMs, the shared activation is grouped per feature for wgrad and the shared weight per input channel for dgrad.
- Int4–int7 change the symmetric code range but still use an int8 container and the INT8 scaled-GEMM path. This sweep measures loss, not packed low-bit memory or throughput.
- W8A8 fuses only the forward GEMM. Dgrad and wgrad each retain one bf16 operand and use quantize → dequantize → bf16 matmul, while W8A16 uses that fallback for every GEMM.
- Companion experiments: `fp8_tensor_dtype/` and `int_granularity/`.

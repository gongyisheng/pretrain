# Integer Tensor-Dtype Sensitivity

Compare Qwen3-51M pretraining in bf16 with rowwise integer quantization from int8 through int4. W8A16 quantizes weights only; W8A8 quantizes weights and activations. `grad_out` remains bf16.

## Hypothesis

W8A16 should stay close to bf16 at int8 and degrade progressively at lower widths. W8A8 measures the added effect of activation quantization.

## Setup

All quantized runs use rowwise scaling and exclude `lm_head`; `grad_out` remains bf16.

| Config | Weight | Activation | grad_out |
|---|---|---|---|
| qwen3_51m_bf16 | bf16 | bf16 | bf16 |
| qwen3_51m_int8_w8a16 | int8 | bf16 | bf16 |
| qwen3_51m_int8_w8a8 | int8 | int8 | bf16 |
| qwen3_51m_int7_w8a16 | int7 | bf16 | bf16 |
| qwen3_51m_int7_w8a8 | int7 | int7 | bf16 |
| qwen3_51m_int6_w8a16 | int6 | bf16 | bf16 |
| qwen3_51m_int6_w8a8 | int6 | int6 | bf16 |
| qwen3_51m_int5_w8a16 | int5 | bf16 | bf16 |
| qwen3_51m_int5_w8a8 | int5 | int5 | bf16 |
| qwen3_51m_int4_w8a16 | int4 | bf16 | bf16 |
| qwen3_51m_int4_w8a8 | int4 | int4 | bf16 |

All runs share: Qwen3 51M, seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps and min_lr=5e-5, bf16 mixed precision, OpenWebText, seed 42, and `eval_steps=100`.

## Run

```bash
nohup bash experiments/int8_tensor_dtype/run.sh > logs/int8_tensor_dtype.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-tensor-dtype`.

| Config | Final Val Loss | Δ vs bf16 |
|---|---|---|
| qwen3_51m_bf16 | | 0 |
| qwen3_51m_int8_w8a16 | | |
| qwen3_51m_int8_w8a8 | | |
| qwen3_51m_int7_w8a16 | | |
| qwen3_51m_int7_w8a8 | | |
| qwen3_51m_int6_w8a16 | | |
| qwen3_51m_int6_w8a8 | | |
| qwen3_51m_int5_w8a16 | | |
| qwen3_51m_int5_w8a8 | | |
| qwen3_51m_int4_w8a16 | | |
| qwen3_51m_int4_w8a8 | | |

## Notes

- `int4`–`int7` set a lower-bit code range but use an int8 tensor container; W8A16 and W8A8 describe which tensors are quantized.
- This measures training-loss sensitivity, not packed low-bit memory or throughput. Everything outside eligible linear layers remains bf16/fp32, and `lm_head` is excluded.

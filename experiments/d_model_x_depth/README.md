# d_model × depth

Test whether allocating a fixed activated non-embedding budget to `d_model` or depth improves validation loss. Here **depth** means `n_layers`; MLP intermediate width stays at `3*d_model`.

## Setup

All configs derive from `configs/qwen3_51m.yaml`, retaining head dimension 64, query/KV ratio two, QK normalization, attention sinks, RoPE, tied embeddings, sequence length 1024, and seed 42. Only the swept architecture fields and checkpoint/W&B identifiers change.

The matched budget is **activated non-embedding parameters**, `TransformerLM.compute_parameters(...)["active_non_emb"]`. It excludes the tied token/output embedding matrix, counted once in total parameters. These dense models activate every non-embedding parameter, so `non_emb = active_non_emb`; total parameters intentionally vary with `d_model`.

The common anchor is `d_model=1024`, eight layers, and MLP intermediate width 3072, with **100,681,856 active non-embedding parameters**. Both experiments use exactly this anchor architecture.

Depths are solved against the anchor budget and rounded to whole layers. Maximum absolute active non-embedding deviation is 0.0637%.

| Config | `d_model` | Layers | MLP width | Heads / KV | `active_non_emb` | Total parameters | Forward FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [qwen3_114m_a101m_d256.yaml](qwen3_114m_a101m_d256.yaml) | 256 | 128 | 768 | 4 / 2 | 100,745,984 (100.746M) | 113,623,808 (113.624M) | 361,620,736 |
| [qwen3_126m_a101m_d512.yaml](qwen3_126m_a101m_d512.yaml) | 512 | 32 | 1,536 | 8 / 4 | 100,700,928 (100.701M) | 126,456,576 (126.457M) | 320,072,192 |
| [qwen3_152m_a101m_d1024.yaml](qwen3_152m_a101m_d1024.yaml) | 1024 | 8 | 3,072 | 16 / 8 | 100,681,856 (100.682M) | 152,193,152 (152.193M) | 337,896,448 |

For `d = d_model`, `m = mlp.intermediate_size`, and `L = n_layers`:

```text
active_non_emb = L*(3*d^2 + 3*d*m + 2*d + 128 + d/64) + d
P_total = active_non_emb + 50304*d
```

Per-layer terms count GQA projections, SwiGLU projections, two RMSNorms, QK norms, and attention sinks; the final `d` counts the output RMSNorm. The padded vocabulary has 50,304 entries.

Counts and forward FLOPs come from `TransformerLM.compute_parameters` and `TransformerLM.compute_flops` at sequence length 1024. Counts also match construction on meta tensors. FLOPs follow the repository's analytic convention, including its unpadded vocabulary for the output projection; they estimate forward cost, not measured training cost.

Base training remains unchanged: batch size 16, gradient accumulation 16, 50,000 steps, bf16, MuonAdam, learning rate 5e-4, and the cosine schedule. Checkpoints occur every 5,000 steps; evaluation every 100 steps uses 100 batches.

Filenames and checkpoint paths use `qwen3_<total>m_a101m_d<d_model>`: the first count is rounded total parameters (114M, 126M, or 152M); `a101m` is rounded activated non-embedding parameters. Exact counts appear above.

## Run

From the repository root, choose a free GPU after checking `nvidia-smi`:

```bash
CUDA_VISIBLE_DEVICES=<free-device> bash experiments/d_model_x_depth/run.sh
```

The runner resolves the repository root, runs all three configs sequentially, forwards additional training arguments unchanged, and stops on failure. Checkpoints use `checkpoints/d_model_x_depth/qwen3_<total>m_a101m_d<d_model>/`. W&B project: `pretrain-d-model-x-depth`; run names: `qwen3-<total>m-a101m-d<d_model>`.

## Results

| `d_model` | Validation loss | Tokens/second | Peak GPU memory | Status |
| ---: | ---: | ---: | ---: | --- |
| 256 | Pending | Pending | Pending | Not run |
| 512 | Pending | Pending | Pending | Not run |
| 1024 | Pending | Pending | Pending | Not run |

## Notes

Matching active non-embedding parameters does not fix total parameters, compute, or memory. The embedding matrix grows with `d_model`; attention and activation costs also vary. Batch size and accumulation stay fixed. The 128-layer model adds substantial sequential execution and activation storage; equal parameter counts do not imply equal throughput. Measure peak memory and throughput alongside loss. No training has been launched.

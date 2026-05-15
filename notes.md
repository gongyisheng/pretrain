# perf-20260515: Qwen3 57M Training Throughput

Goal: improve `python benchmarks/bench_train.py --config configs/qwen3_57m.yaml`
on a single CUDA device (cuda:0).

Hardware: NVIDIA GeForce RTX 5060 Ti (sm_120, Blackwell, 16 GB).
Software: PyTorch 2.10.0+cu128.

## Config under test

- arch: qwen3, 8 layers, 8 heads, 4 kv heads, d_model=512, vocab=50257
- seq_len=1024, bf16, qk_norm=true, activation_checkpointing=false
- effective batch = 256 examples

## Baseline (main)

```
20 steps in 69.3s (after 5 warmup)
Tokens/sec: 75,550 (±0.1% across 3 repeats)
```

## Variants tested

| # | Variant | tok/s | Δ | Decision |
|---|---|---|---|---|
| 0 | baseline (bs=8, ga=32) | 75,550 | — | — |
| 1 | SDPA `enable_gqa=True` | 60,621 | **-19.8%** | revert (regression) |
| 2 | qk-norm drops reshape + RoPE gather fused + prefetch=4 | 75,559 | -0.0% | net-neutral |
| 3 | `mode="reduce-overhead"` (CUDA graphs) | crash | — | revert; trainer loop needs deeper retrofit |
| 4 | Force flash-only SDPA backend | 75,608 | -0.1% | net-neutral; inductor already picks flash |
| 5 | **Liger fused linear+CE** | 37,995 | **-49.7%** | revert (regression) |
| 6 | **bs=16, ga=16 (eff batch unchanged)** | 77,966 | +3.2% | superseded by #7 |
| 7 | **bs=32, ga=8 (eff batch unchanged)** | **78,914** | **+4.5%** | **commit** |
| 8 | bs=64, ga=4 | OOM | — | doesn't fit in 16 GB |
| 9 | `mode="reduce-overhead"` + `cudagraph_mark_step_begin` + bs=32 | OOM | — | CUDA graph buffers don't fit |
| 10 | num_workers=8 | 78,899 | 0.0% | data isn't the bottleneck |
| 11 | num_workers=4, prefetch_factor=4 (with bs=32/ga=8) | 78,900 | 0.0% | data isn't the bottleneck |

All numeric variants produced bit-identical loss curves through step 25 → no
training-dynamics regression.

## What worked

**Re-balancing `batch_size` / `gradient_accumulation_steps`** is the only
change that moved throughput. bs=8 → bs=32 (with ga adjusted from 32 → 8 to
keep effective batch at 256) gives +4.5%. Math is unchanged (same number of
examples averaged into the gradient); the win is amortizing kernel-launch,
autocast, and dataloader-handoff overhead over 4× as many tokens per
micro-step. Capped at bs=32 by the 16 GB memory budget — bs=64 OOMs.

## What didn't work — and why it's interesting

- **Liger fused linear+CE was a 50% regression.** Counterintuitive: avoiding
  the (B·S, V) ≈ 825 MB bf16 logits tensor should help. It doesn't here
  because (a) RTX 5060 Ti at this batch size isn't memory-bound on the LM
  head; (b) Liger's Triton kernel sits *outside* `torch.compile`'s graph, so
  we lose inductor's fusion of `lm_head_linear → CE`; (c) the chunked Triton
  kernel adds per-chunk launch overhead that the model is small enough to
  feel. The win shifts in the opposite direction for larger models /
  smaller compute-to-memory ratios; keep this in mind for the 0.5B config.

- **`enable_gqa=True` was a 20% regression.** PyTorch's internal GQA
  broadcasting in SDPA is slower than the manual `expand→reshape` we do
  before flash-attn on sm_120 / 2.10. Manual expansion stays.

- **CUDA graphs (`mode="reduce-overhead"`) didn't survive contact** with the
  trainer's micro-step loop even with `cudagraph_mark_step_begin`. There's
  cross-iteration tensor reuse (prefetched batches, `record_stream` calls,
  re-bound `input_ids`/`position_ids`) that the graph capturer flags as
  unsafe. A real fix would clone all model inputs and rework the prefetch
  stream interaction — non-trivial; left for a dedicated branch.

- **qk-norm 4D normalization, RoPE gather fusion, prefetch_factor, flash
  backend force, num_workers tweaks** — all genuine code/config refactors
  but all net-zero throughput. `torch.compile` is already fusing or
  inductor's default backend selection is already optimal at this scale.

## What we're committing

Only the `qwen3_57m.yaml` config change: `batch_size: 8 → 32`,
`gradient_accumulation_steps: 32 → 8`. Same effective batch (256). +4.5%.

## Where the real wins likely live (out of scope here)

- **Fused linear+CE that lives inside `torch.compile`.** A
  `@torch.compile`'d custom CE that takes hidden+weight+targets could win
  back what Liger loses — let inductor fuse the linear with the
  softmax/log_softmax reduction. Worth ~10% at 0.5B-class configs.
- **CUDA graphs with a refactored micro-step loop.** Likely the single
  largest remaining lever (~5-15%) on small-model / launch-bound workloads
  like this one. Needs trainer surgery: clone inputs, eliminate
  `record_stream` interactions, decouple eval-path compiled graph.
- **Use a GPU with more SMs.** `max-autotune-gemm` is disabled here ("Not
  enough SMs"); on a larger GPU the matmul autotune alone usually buys
  5-10%.

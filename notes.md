# perf-20260515: Qwen3 57M Training Throughput

Goal: improve `python benchmarks/bench_train.py --config configs/qwen3_57m.yaml`
on a single CUDA device (cuda:0).

Hardware: NVIDIA GeForce RTX 5060 Ti (sm_120, Blackwell). PyTorch 2.10.0+cu128.

## Config under test

- arch: qwen3, 8 layers, 8 heads, 4 kv heads, d_model=512, vocab=50257
- seq_len=1024, batch_size=8, grad_accum=32, bf16
- qk_norm=true, activation_checkpointing=false

## Ideas (from full code review)

### High-impact
1. **SDPA native GQA** (`attention.py:107-108`): drop manual expand+reshape, use
   `F.scaled_dot_product_attention(..., enable_gqa=True)`.
2. **Fused cross-entropy** (`trainer.py:24`): avoid materializing `(B*S, V)` logits.
3. **Rebalance batch / grad-accum**: bs=8/ga=32 → bs=32/ga=8 to reduce launches.
4. **RoPE indexing** (`pos_emb.py:62-63`, TODO'd): fold the per-call gather and
   dtype cast into the compiled `_apply_rope` so inductor can fuse them.

### Medium-impact
5. **Drop reshape around qk-norm** (`attention.py:47-48, 98-99`): apply RMSNorm
   to the 4D tensor directly — same math, no contiguous copy.
6. **Bump DataLoader `prefetch_factor`** (`trainer.py:113, 119`): 1 → 4.
7. **`torch.compile(..., mode="max-autotune")`** or **`mode="reduce-overhead"`**
   (CUDA graphs).
8. **Skip embedding dropout when p==0**.

### Low-impact / cleanup
9. CUDA graphs for the inner micro-step.
10. Keep accumulated loss on GPU; only sync at `log_every`.
11. Drop nested `@torch.compile` on inner ops now that the model is compiled.
12. **Force flash-attention SDPA backend**: on Blackwell, the default backend
    selection can pick cuDNN attention; eager microbench showed cuDNN was ~150×
    slower than flash. (Inside the compiled graph this turned out to be a no-op
    — inductor already picks flash.)

## Baseline

```
20 steps in 69.3s (after 5 warmup)
Tokens/sec: 75,652
```

## Results

| Variant | tok/s | Δ vs baseline | Decision |
|---|---|---|---|
| baseline (main) | 75,652 | — | — |
| #1 enable_gqa=True | 60,621 | **-19.9%** | revert (regression) |
| #4 RoPE gather fused + #5 qk-norm 4D + #6 prefetch=4 | 75,559 | -0.1% | net-neutral |
| #7 reduce-overhead (CUDA graphs) | crashed | — | revert (needs `cudagraph_mark_step_begin` in trainer loop) |
| #12 flash SDPA forced (excluding cuDNN) | 75,608 | -0.1% | net-neutral |

All variants produced bit-identical loss values (10.1176 at step 25) → no numerics
regression. Variance across repeated runs is ~±0.1%.

## What we learned

1. **`enable_gqa=True` is a measurable regression on this GPU/PyTorch combo**
   (~20% slower). The manual `expand+reshape` path is faster than SDPA's
   internal head broadcasting here. Don't apply this on sm_120 / PyTorch 2.10.
2. **The qk-norm reshape, RoPE-gather fusion, and prefetch_factor changes are
   all net-zero on this config.** `torch.compile` (default mode) appears to
   already fuse the gather + cast, and the qk-norm contiguous copy is either
   fused away or too small to measure at d_head=64, batch=8.
3. **`mode="reduce-overhead"` is incompatible with the current grad-accum loop**:
   CUDA graphs flag the input tensor reuse pattern as unsafe. To use it the
   trainer would need to call `torch.compiler.cudagraph_mark_step_begin()`
   between micro-steps and/or clone inputs. Real but bigger refactor.
4. **Inductor already selects flash attention** inside the compiled model on
   this GPU, even though eager microbench picks the much-slower cuDNN backend.
   Forcing the backend via `sdpa_kernel` only matters in eager paths.
5. **The single-precision (fp32) warning** seen at startup (`Mismatch dtype
   between input and weight ... Cannot dispatch to fused implementation`)
   is **not** from the qk-norm reshape — it persists after that change.
   Likely originates from RMSNorm with fp32 weight under bf16 autocast.
   Worth a follow-up but unrelated to the current bottleneck.

## Decision

None of the safe local refactors moved throughput. Per the rule "commit if it
improves, otherwise revert", all changes are reverted on this branch.

## Where the real wins likely live (out of scope for this attempt)

- **Fused linear+CE for the LM head** (#2). LM head materialises a
  ~825 MB bf16 logits tensor at this config — chunked/fused CE typically pays
  off most for small models with relatively large vocabs.
- **bs=32/ga=8 (or bs=64/ga=4)** (#3). Effective batch unchanged; per-step
  launch + accumulation overhead goes down. Likely needs #2 first for memory
  headroom.
- **CUDA graphs via `reduce-overhead`** (#7) once the grad-accum loop is
  retrofitted with `cudagraph_mark_step_begin`. Largest single-lever speedup
  on small models.

These each warrant their own branch with parity verification (loss curve match
over a few hundred steps) before merging.

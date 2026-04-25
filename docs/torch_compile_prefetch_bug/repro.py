"""Minimal standalone reproducer: torch.compile + CUDA prefetch-stream race.

See `docs/torch_compile_prefetch_bug/README.md` for full background, root
cause, and fix.

Quick reference
---------------
    # All three modes side by side: BAD vs GOOD vs FIX
    python docs/torch_compile_prefetch_bug/repro.py --steps 500 --mode all

    # Single mode:
    python docs/torch_compile_prefetch_bug/repro.py --steps 500 --mode prefetch
    python docs/torch_compile_prefetch_bug/repro.py --steps 500 --mode no-prefetch
    python docs/torch_compile_prefetch_bug/repro.py --steps 500 --mode record-stream

    # Disabling torch.compile globally also fixes the bug:
    TORCHDYNAMO_DISABLE=1 python docs/torch_compile_prefetch_bug/repro.py \
        --steps 500 --mode prefetch

Data
----
Uses `data/train.bin` (uint16 token ids) if present. The synthetic
noisy-bigram fallback does NOT trigger the bug — realistic Zipfian token
distributions appear to be required for the gradient bias to compound
enough to land in a different basin. To run externally, point `--data` at
any tokenized pretraining .bin file (e.g. a nanoGPT openwebtext shard).

Tested on
---------
NVIDIA RTX 5090, torch 2.10.0+cu128, CUDA 12.8, bf16 autocast.
"""
from __future__ import annotations
import argparse
import contextlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_nullcontext = contextlib.nullcontext


# ---------------------------------------------------------------------------
# Per-op fused kernels (mirrors the project's structure: each kernel is
# `@torch.compile`-decorated as a module-level function, then the whole
# model is wrapped in another `torch.compile(...)`).
# ---------------------------------------------------------------------------

@torch.compile
def fused_rmsnorm(x, weight, eps):
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


@torch.compile
def fused_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


@torch.compile
def fused_swiglu(gate, up):
    return F.silu(gate) * up


@torch.compile
def fused_flash_attn(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


@torch.compile
def fused_cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Model: Qwen3-57M-style (GQA + RoPE + RMSNorm + SwiGLU, weight-tied LM head)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        orig_shape = x.shape
        out = fused_rmsnorm(x.reshape(-1, orig_shape[-1]), self.weight, self.eps)
        return out.reshape(orig_shape)


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2) / d_head))
        pos = torch.arange(max_seq_len)
        ang = pos[:, None] * freqs[None, :]
        ang = torch.cat([ang, ang], dim=-1)
        self.register_buffer("cos", torch.cos(ang))
        self.register_buffer("sin", torch.sin(ang))

    def forward(self, x):  # x: (B, H, S, D)
        S = x.shape[2]
        cos = self.cos[:S][None, None, :, :].to(x.dtype)
        sin = self.sin[:S][None, None, :, :].to(x.dtype)
        return fused_rope(x, cos, sin)


class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)

    def forward(self, x, rope):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.dh).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv, self.dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv, self.dh).transpose(1, 2)
        q = self.q_norm(q.reshape(-1, S, self.dh)).view(B, self.n_heads, S, self.dh)
        k = self.k_norm(k.reshape(-1, S, self.dh)).view(B, self.n_kv, S, self.dh)
        q, k = rope(q), rope(k)
        k = k[:, :, None].expand(B, self.n_kv, self.n_groups, S, self.dh).reshape(B, self.n_heads, S, self.dh)
        v = v[:, :, None].expand(B, self.n_kv, self.n_groups, S, self.dh).reshape(B, self.n_heads, S, self.dh)
        out = fused_flash_attn(q, k, v, 1.0 / self.dh ** 0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hd):
        super().__init__()
        self.gate_proj = nn.Linear(d, hd, bias=False)
        self.up_proj = nn.Linear(d, hd, bias=False)
        self.down_proj = nn.Linear(hd, d, bias=False)

    def forward(self, x):
        return self.down_proj(fused_swiglu(self.gate_proj(x), self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv, hd):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hd)

    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.ffn(self.ln2(x))
        return x


class Qwen3Mini(nn.Module):
    def __init__(self, vocab=50304, n_layers=8, d_model=512, n_heads=8, n_kv=4, hd=2048, S_max=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.rope = RoPE(d_model // n_heads, S_max)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, n_kv, hd) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h, self.rope)
        return self.head(self.ln_f(h))


# ---------------------------------------------------------------------------
# Data loading: CPU pinned memory, async H2D copy on a side CUDA stream.
# This is what triggers the bug under torch.compile.
# ---------------------------------------------------------------------------

def load_data(data_path: str | None, n_total_samples: int, seq_len: int,
              vocab: int, seed: int) -> torch.Tensor:
    """Returns a CPU pinned tensor of shape (N, S+1) with token ids."""
    if data_path:
        import numpy as np
        arr = np.memmap(data_path, dtype=np.uint16, mode="r")
        n_avail = len(arr) // seq_len - 1
        n = min(n_total_samples, n_avail)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_avail, generator=g)[:n].numpy()
        out = np.empty((n, seq_len + 1), dtype=np.int64)
        for i, ii in enumerate(idx):
            out[i] = np.asarray(arr[ii * seq_len : ii * seq_len + seq_len + 1]).astype(np.int64)
        t = torch.from_numpy(out)
    else:
        # Noisy-bigram synthetic fallback (does NOT trigger bug for us; included
        # so the script runs without a tokenized corpus).
        g_cpu = torch.Generator().manual_seed(seed)
        f = torch.randperm(vocab, generator=g_cpu)
        rand_tok = torch.randint(0, vocab, (n_total_samples, seq_len), generator=g_cpu)
        keep_det = torch.rand(n_total_samples, seq_len, generator=g_cpu) < 0.5
        t = torch.empty(n_total_samples, seq_len + 1, dtype=torch.long)
        t[:, 0] = torch.randint(0, vocab, (n_total_samples,), generator=g_cpu)
        for i in range(seq_len):
            det = f[t[:, i]]
            t[:, i + 1] = torch.where(keep_det[:, i], det, rand_tok[:, i])
    return t.pin_memory()


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def train(*, batch_size: int, ga: int, n_steps: int, data_path: str | None,
          use_prefetch_stream: bool, compute_on_prefetch_stream: bool = False,
          use_record_stream: bool = False,
          warmup_steps: int = 1500, seed: int = 42,
          lr: float = 6e-4, log_every: int = 25, vocab: int = 50304,
          seq_len: int = 1024) -> list[tuple[int, float]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Qwen3Mini(vocab=vocab, S_max=seq_len).cuda()
    model = torch.compile(model)

    decay, no_decay = [], []
    for nm, p in model.named_parameters():
        if p.ndim < 2 or "ln" in nm or "bias" in nm or "norm" in nm:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=True,
    )

    cpu_data = load_data(data_path, n_steps * batch_size * ga + batch_size,
                         seq_len, vocab, seed=seed)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, t):
            self.t = t
        def __len__(self):
            return len(self.t)
        def __getitem__(self, i):
            row = self.t[i]
            return row[:-1].clone(), row[1:].clone()

    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        _DS(cpu_data), batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, generator=g,
    )
    print(f"  data: shape={tuple(cpu_data.shape)} (using real DataLoader, pin_memory=True)")
    data_iter = iter(loader)

    def _next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    prefetch_stream = torch.cuda.Stream() if use_prefetch_stream else None
    losses = []
    t0 = time.time()

    def fetch_to_gpu(x_cpu, y_cpu):
        """Async H2D copy via prefetch stream (or sync if not using stream)."""
        if prefetch_stream is not None:
            with torch.cuda.stream(prefetch_stream):
                x = x_cpu.to("cuda", non_blocking=True)
                y = y_cpu.to("cuda", non_blocking=True)
        else:
            x = x_cpu.to("cuda")
            y = y_cpu.to("cuda")
        return x, y

    for step in range(n_steps):
        cur_lr = lr * min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else lr
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)

        # Prefetch the first micro-batch
        x_cpu, y_cpu = _next_batch()
        x, y = fetch_to_gpu(x_cpu, y_cpu)

        accum_t = torch.zeros(1, device="cuda")
        for mi in range(ga):
            if prefetch_stream is not None and not compute_on_prefetch_stream:
                # Wait for the H2D copy of the current (x,y) to finish before
                # the default stream uses them.
                torch.cuda.current_stream().wait_stream(prefetch_stream)

                # Optional: tell the caching allocator that x/y are in use on
                # the current stream, so their storage isn't freed when the
                # prefetch_stream's only enqueued op (the H2D copy) finishes.
                # This is the documented "best practice" for cross-stream
                # tensor handoff.
                if use_record_stream:
                    x.record_stream(torch.cuda.current_stream())
                    y.record_stream(torch.cuda.current_stream())

            # Kick off prefetch of the next micro-batch in parallel with compute
            if mi < ga - 1:
                next_x_cpu, next_y_cpu = _next_batch()
                next_x, next_y = fetch_to_gpu(next_x_cpu, next_y_cpu)

            # Optional: run the compiled model on the SAME stream as the H2D
            # copy. With producer (H2D) and consumer (compute) on one stream,
            # CUDA's natural in-stream ordering supplies the synchronization
            # and no cross-stream wait is needed.
            ctx = (torch.cuda.stream(prefetch_stream)
                   if compute_on_prefetch_stream and prefetch_stream is not None
                   else _nullcontext())
            with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = fused_cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)) / ga
                loss.backward()
                accum_t = accum_t + loss.detach()

            if mi < ga - 1:
                x, y = next_x, next_y

        # If we ran compute on the prefetch stream, sync back to the default
        # stream before the optimizer touches grads on the default stream.
        if compute_on_prefetch_stream and prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % log_every == 0 or step == 0:
            losses.append((step + 1, accum_t.item()))

        # Match the project's eval cadence: every 100 steps run a few forward
        # passes in eval mode. _generate_sample iterates with varying seq
        # lengths (1, 2, ..., 50), each new shape triggering torch.compile
        # recompilation -- which amplifies the bug.
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                idx = torch.zeros((1, 1), dtype=torch.long, device="cuda")
                for _ in range(50):
                    _ = model(idx[:, -seq_len:])
                    idx = torch.cat([idx, torch.zeros_like(idx[:, :1])], dim=1)
            model.train()

    print(f"  total: {time.time() - t0:.1f}s")
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=str, default="data/train.bin")
    ap.add_argument("--mode",
                    choices=["both", "prefetch", "no-prefetch", "same-stream",
                             "record-stream", "all"],
                    default="both",
                    help="which run(s) to perform. 'same-stream' runs the "
                         "compiled model on the same stream as the H2D copy. "
                         "'record-stream' keeps the cross-stream design but "
                         "calls .record_stream() on the H2D'd tensors. "
                         "'all' runs every mode for comparison.")
    args = ap.parse_args()
    data_path = args.data or None

    print("=== torch.compile + CUDA prefetch-stream divergence repro ===")
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"config: batch_size={args.batch_size}  ga={args.ga}  steps={args.steps}  "
          f"seed={args.seed}  data={data_path or 'synthetic'}\n")

    runs: dict[str, list[tuple[int, float]]] = {}
    if args.mode in ("both", "prefetch", "all"):
        print("[1] compiled + prefetch_stream + compute on default stream (BAD path)")
        runs["prefetch"] = train(batch_size=args.batch_size, ga=args.ga,
                                 n_steps=args.steps, data_path=data_path,
                                 use_prefetch_stream=True, seed=args.seed)
    if args.mode in ("both", "no-prefetch", "all"):
        print("\n[2] compiled, NO prefetch stream (GOOD path)")
        runs["no-prefetch"] = train(batch_size=args.batch_size, ga=args.ga,
                                    n_steps=args.steps, data_path=data_path,
                                    use_prefetch_stream=False, seed=args.seed)
    if args.mode == "same-stream":
        # NOTE: this mode currently crashes — running the compiled forward+
        # backward inside `with torch.cuda.stream(prefetch_stream):` while
        # opt.step() / accum tensors live on the default stream produces a
        # caching-allocator event-recording fault. Kept opt-in so it's
        # easy to re-test if/when fixed; not included in 'all'.
        print("\n[3] compiled + H2D and compute on the SAME stream (CRASHES — see note)")
        runs["same-stream"] = train(batch_size=args.batch_size, ga=args.ga,
                                    n_steps=args.steps, data_path=data_path,
                                    use_prefetch_stream=True,
                                    compute_on_prefetch_stream=True,
                                    seed=args.seed)
    if args.mode in ("record-stream", "all"):
        print("\n[4] compiled + prefetch_stream + record_stream() (candidate fix)")
        runs["record-stream"] = train(batch_size=args.batch_size, ga=args.ga,
                                      n_steps=args.steps, data_path=data_path,
                                      use_prefetch_stream=True,
                                      use_record_stream=True,
                                      seed=args.seed)

    # Print whichever runs we collected, side by side.
    keys = list(runs.keys())
    if len(keys) >= 2:
        n_steps_logged = len(runs[keys[0]])
        header = f"{'step':>5s}  " + "  ".join(f"{k:>14s}" for k in keys)
        print()
        print(header)
        for i in range(n_steps_logged):
            row_step = runs[keys[0]][i][0]
            line = f"{row_step:>5d}  " + "  ".join(f"{runs[k][i][1]:>14.4f}" for k in keys)
            print(line)

        # Mean over the second half (warmup mostly past)
        half = n_steps_logged // 2
        means = {k: sum(v[1] for v in runs[k][half:]) / (n_steps_logged - half) for k in keys}
        baseline_key = "no-prefetch" if "no-prefetch" in keys else keys[0]
        print("\nMean loss over second half (lower = better):")
        for k in keys:
            delta = means[k] - means[baseline_key]
            tag = " (baseline)" if k == baseline_key else f" (gap vs {baseline_key}: {delta:+.4f})"
            print(f"  {k:>14s}: {means[k]:.4f}{tag}")
    else:
        print(f"\n{'step':>5s}  {'loss':>9s}")
        only_run = next(iter(runs.values()))
        for s, ls in only_run:
            print(f"{s:>5d}  {ls:>9.4f}")


if __name__ == "__main__":
    main()

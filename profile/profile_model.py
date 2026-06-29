"""Standalone forward/backward kernel profiler for nsys / ncu.

Drives any configs/*.yaml with a fixed synthetic batch (no Trainer, no
dataloader) so shapes stay constant across iterations — clean fwd/bwd
isolation and stable kernels for ncu. NVTX ranges label forward/backward;
cudaProfilerApi gates the capture to steady-state steps (post-warmup).

Profiles the compiled bf16 path by default (what training runs). --no-compile
gives eager aten attribution (you see _grouped_mm, sdpa, scatter/gather directly).

nsys:
    nsys profile --capture-range=cudaProfilerApi -o logs/profiles/qwen3_moe \
        python profile/profile_model.py --config configs/qwen3_183m_a51m.yaml
    nsys stats --report cuda_gpu_kern_sum logs/profiles/qwen3_moe.nsys-rep

ncu (slow — use --steps 1):
    ncu --profile-from-start off --nvtx --nvtx-include "iter/" \
        -o logs/profiles/qwen3_moe_ncu \
        python profile/profile_model.py --config configs/qwen3_183m_a51m.yaml --steps 1
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")

from src.model import build_model
from src.training.loss import compute_loss
from src.utils.config import load_config
from src.utils.masking_utils import build_causal_attention_mask

_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    B = args.batch_size or config.training.batch_size
    S = args.seq_len or config.max_seq_len
    vocab_size = config.model.vocab_size
    mp = config.training.mixed_precision
    use_amp = mp != "no"
    amp_dtype = _AMP_DTYPE.get(mp, torch.bfloat16)

    model = build_model(config).to(device).train()
    if not args.no_compile:
        # Readable Triton kernel names for ncu/nsys attribution.
        torch._inductor.config.triton.unique_kernel_names = True
        model = torch.compile(model)

    # One synthetic batch, reused every iter so shapes never change.
    input_ids = torch.randint(0, vocab_size, (B, S), device=device)
    labels = torch.randint(0, vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    attn_mask = build_causal_attention_mask(
        B,
        S,
        device,
        attn_implementation=config.model.attn_kwargs["attn_implementation"],
    )
    aux_coef = config.model.mlp_kwargs.get("aux_loss_coef", 0.0)

    def step():
        torch.cuda.nvtx.range_push("forward")
        with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
            logits, aux_loss = model(
                input_ids, position_ids=position_ids, attn_mask=attn_mask
            )
            loss = compute_loss(
                logits,
                labels,
                config.training.loss_fn,
                label_smoothing=config.training.label_smoothing,
            )
            if aux_loss is not None:
                loss = loss + aux_coef * aux_loss
        torch.cuda.nvtx.range_pop()
        if not args.forward_only:
            torch.cuda.nvtx.range_push("backward")
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.nvtx.range_pop()

    print(
        f"[profile] {args.config} | B={B} S={S} | compile={not args.no_compile} | "
        f"{'fwd' if args.forward_only else 'fwd+bwd'} | warmup={args.warmup} steps={args.steps}"
    )

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.steps):
        torch.cuda.nvtx.range_push(f"iter/{i}")
        step()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[profile] done")


if __name__ == "__main__":
    main()

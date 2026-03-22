"""Per-weight statistics for a single checkpoint.

Usage:
    python scripts/debug_weight_stats.py --ckpt <checkpoint.pt>
    python scripts/debug_weight_stats.py --ckpt checkpoints/attn_res/gpt2_d512_l12/step_10000.pt
    python scripts/debug_weight_stats.py --ckpt checkpoints/attn_res/gpt2_d512_l12/step_10000.pt --sort max --top 30

Outputs max / min / mean / std for every weight tensor, sortable by any stat.
"""
import argparse
import torch


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_stats(t: torch.Tensor) -> dict:
    f = t.float().flatten()
    nan_mask = torch.isnan(f)
    valid = f[~nan_mask]
    nan_count = nan_mask.sum().item()
    return {
        "max":   valid.max().item() if valid.numel() > 0 else float("nan"),
        "min":   valid.min().item() if valid.numel() > 0 else float("nan"),
        "mean":  valid.mean().item() if valid.numel() > 0 else float("nan"),
        "std":   valid.std().item() if valid.numel() > 0 else float("nan"),
        "nans":  nan_count,
        "numel": f.numel(),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-weight max/min/mean/std for a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--sort", default="std", choices=["max", "min", "mean", "std", "nans", "name"],
                        help="Sort by this stat (default: std)")
    parser.add_argument("--top",  type=int, default=0,
                        help="Show only top N rows (default: 0 = all)")
    parser.add_argument("--abs",  action="store_true",
                        help="Sort by absolute value of the stat")
    args = parser.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = load(args.ckpt)
    step = ckpt.get("step", "?")
    model_sd = ckpt.get("model", ckpt)  # support bare state_dict too

    rows = []
    for name, tensor in model_sd.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        s = tensor_stats(tensor)
        rows.append((name, s))

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        key_fn = (lambda r: abs(r[1][args.sort])) if args.abs else (lambda r: r[1][args.sort])
        rows.sort(key=key_fn, reverse=(args.sort != "min"))

    display = rows[:args.top] if args.top > 0 else rows

    print(f"\nStep: {step}  |  {len(rows)} float tensors  |  sorted by {'|' + args.sort + '|' if args.abs else args.sort}\n")
    hdr = f"  {'parameter':<55} {'max':>11} {'min':>11} {'mean':>11} {'std':>11}  {'nans':>6}  {'numel':>8}"
    print(hdr)
    print("  " + "-" * 124)
    for name, s in display:
        nan_flag = "  !" if s["nans"] > 0 else ""
        print(f"  {name:<55} {s['max']:11.5f} {s['min']:11.5f} {s['mean']:11.5f} {s['std']:11.5f}  {s['nans']:>6}{nan_flag}  {s['numel']:>8,}")

    if args.top > 0 and args.top < len(rows):
        print(f"\n  ... ({len(rows) - args.top} more rows hidden, use --top 0 to show all)")

    print()


if __name__ == "__main__":
    main()

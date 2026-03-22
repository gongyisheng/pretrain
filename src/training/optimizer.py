import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from src.utils.config import TrainConfig


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
# Background: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon — MomentUm Orthogonalized by Newton-schulz.

    Applies a Newton-Schulz orthogonalization step to matrix-shaped gradients
    before the momentum update, which avoids the need for a large Adam eps or
    weight decay on the transformer weight matrices.

    Only suitable for 2-D weight matrices (e.g. Linear layers in transformer
    blocks). Use Adam/AdamW for embeddings, biases, and scalar params.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        backend_steps: int = 5,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# OPTIMIZER BUILDING
# -----------------------------

def _partition_params(model: nn.Module):
    """Split params into (embed, matrix, scalar) for Muon setup.

    - embed:  Embedding layer weights → Adam (their 2-D shape is a lookup table)
    - matrix: all remaining 2-D weights (Linear layers) → Muon
    - scalar: biases, LN weights, and other 1-D params → Adam
    """
    seen_ids: set = set()
    embed_params, matrix_params, scalar_params = [], [], []

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    embed_params.append(p)

    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        if p.ndim >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    return embed_params, matrix_params, scalar_params


def build_optimizers(model: nn.Module, config: TrainConfig) -> list:
    """Return a list of optimizers.

    - adamw: ``[AdamW]``
    - muon:  ``[Muon (matrix params), Adam (embed), Adam (scalar)]``

    Each param group carries a ``base_lr`` key used by :class:`CosineWarmupScheduler`
    to scale the LR proportionally across all optimizers.
    """
    cfg = config.optimizer

    if cfg.name == "muon":
        embed_lr = cfg.muon_embed_lr if cfg.muon_embed_lr > 0 else cfg.lr
        scalar_lr = cfg.muon_scalar_lr if cfg.muon_scalar_lr > 0 else cfg.lr

        embed_params, matrix_params, scalar_params = _partition_params(model)

        opt_muon = Muon(
            matrix_params,
            lr=cfg.lr,
            momentum=cfg.muon_momentum,
            backend_steps=cfg.muon_backend_steps,
        )
        for pg in opt_muon.param_groups:
            pg["base_lr"] = cfg.lr

        opt_embed = torch.optim.Adam(
            [{"params": embed_params, "lr": embed_lr, "base_lr": embed_lr}],
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )
        opt_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )
        return [opt_muon, opt_embed, opt_scalar]

    # AdamW (default)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay, "lr": cfg.lr, "base_lr": cfg.lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": cfg.lr, "base_lr": cfg.lr},
    ]
    return [torch.optim.AdamW(param_groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)]


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Backward-compat wrapper — returns the first optimizer from build_optimizers."""
    return build_optimizers(model, config)[0]


# -----------------------------
# LR SCHEDULER
# -----------------------------

class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr.

    Supports multiple optimizers (e.g., Muon's three-optimizer setup). Each
    param group's LR is scaled from its ``base_lr`` using the same cosine
    curve, so optimizers with different peak LRs decay proportionally.
    """

    def __init__(
        self,
        optimizers,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        max_lr: float,
    ):
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        self.optimizers = optimizers
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

        for opt in self.optimizers:
            for pg in opt.param_groups:
                if "base_lr" not in pg:
                    pg["base_lr"] = pg["lr"]

    def get_scale(self) -> float:
        """LR scale in [min_lr/max_lr, 1.0] for the current step."""
        ratio = self.min_lr / self.max_lr if self.max_lr > 0 else 0.0
        step = self.current_step
        if step < self.warmup_steps:
            return ratio + (1.0 - ratio) * step / max(self.warmup_steps, 1)
        if step >= self.max_steps:
            return ratio
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress))

    def step(self):
        self.current_step += 1
        scale = self.get_scale()
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * scale

    def get_lr(self) -> float:
        return self.optimizers[0].param_groups[0]["lr"]

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.max_steps = state_dict.get("max_steps", self.max_steps)
        self.min_lr = state_dict.get("min_lr", self.min_lr)
        self.max_lr = state_dict.get("max_lr", self.max_lr)


def build_scheduler(optimizers, config: TrainConfig) -> CosineWarmupScheduler:
    """Build a cosine LR scheduler. ``optimizers`` may be a list or a single optimizer."""
    return CosineWarmupScheduler(
        optimizers=optimizers,
        warmup_steps=config.scheduler.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.scheduler.min_lr,
        max_lr=config.optimizer.lr,
    )

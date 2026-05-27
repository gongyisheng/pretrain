import math
import re
import torch
from src.utils.config import TrainConfig


AdamWOptimizer = torch.optim.AdamW


class LionOptimizer(torch.optim.Optimizer):
    """
    Lion: EvoLved Sign Momentum (Chen et al. 2023, https://arxiv.org/pdf/2302.06675).

    Per-coordinate update is `sign(β1·m + (1-β1)·g)` plus decoupled wd, scaled by lr.
    State is a single momentum buffer `exp_avg` per param (half of AdamW).

    Lion maintains one running statistic:
    - m (momentum): EMA of the gradient.

    Update per step:
    c = β1·m + (1-β1)·g                   # interpolated gradient
    θ ← θ - lr · sign(c) - lr · wd · θ
    m = β2·m + (1-β2)·g                   # m updated AFTER use (slower decay)
    """

    def __init__(
        self,
        params,
        lr: float,
        betas=(0.9, 0.99),
        weight_decay: float = 0.0,
        foreach: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        defaults = dict(
            lr=lr, betas=tuple(betas), weight_decay=weight_decay, foreach=foreach
        )
        super().__init__(params, defaults)

    @staticmethod
    def _single_tensor_lion(params, grads, exp_avgs, *, lr, beta1, beta2, wd):
        for p, g, m in zip(params, grads, exp_avgs):
            update = m.mul(beta1).add_(g, alpha=1 - beta1).sign_()
            p.mul_(1 - lr * wd).add_(update, alpha=-lr)
            m.mul_(beta2).add_(g, alpha=1 - beta2)

    @staticmethod
    def _multi_tensor_lion(params, grads, exp_avgs, *, lr, beta1, beta2, wd):
        # foreach kernels require uniform device + dtype per call; group accordingly.
        grouped: dict = {}
        for p, g, m in zip(params, grads, exp_avgs):
            key = (p.device, p.dtype)
            if key not in grouped:
                grouped[key] = ([], [], [])
            grouped[key][0].append(p)
            grouped[key][1].append(g)
            grouped[key][2].append(m)

        for p_list, g_list, m_list in grouped.values():
            update = torch._foreach_mul(m_list, beta1)
            torch._foreach_add_(update, g_list, alpha=1 - beta1)
            torch._foreach_sign_(update)

            torch._foreach_mul_(p_list, 1 - lr * wd)
            torch._foreach_add_(p_list, update, alpha=-lr)

            torch._foreach_mul_(m_list, beta2)
            torch._foreach_add_(m_list, g_list, alpha=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            foreach = group.get("foreach", True)

            params, grads, exp_avgs = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avgs.append(state["exp_avg"])

            if not params:
                continue

            fn = self._multi_tensor_lion if foreach else self._single_tensor_lion
            fn(params, grads, exp_avgs, lr=lr, beta1=beta1, beta2=beta2, wd=wd)

        return loss


def build_optimizer(
    model: torch.nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied only to non-bias, non-layernorm params.

    Each key in `config.optimizer.lr_mult` is a regular expression. A parameter
    is assigned to the first pattern (in dict insertion order) that
    `re.search`-matches its name; matched params go into their own AdamW group
    with that multiplier. Params matching no pattern fall into the default
    groups with `lr_mult=1.0`. Within every group, weight decay still follows
    the standard rule (`ndim >= 2` and no `"ln"`/`"bias"` in the name → decay).

    Tied mode: `lm_head.weight is token_emb.weight`, so the param is enumerated
    only under `token_emb.weight` — an `lm_head` pattern in `lr_mult` is a
    silent no-op.
    """
    patterns = [(re.compile(k), k) for k in config.optimizer.lr_mult.keys()]
    wd = config.optimizer.weight_decay
    groups: dict = {}  # (matched_pattern_key_or_None, is_no_decay) -> [params]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = None
        for rx, key in patterns:
            if rx.search(name):
                matched = key
                break
        is_no_decay = param.ndim < 2 or "ln" in name or "bias" in name
        groups.setdefault((matched, is_no_decay), []).append(param)

    param_groups = []
    for (matched, is_no_decay), params in groups.items():
        mult = config.optimizer.lr_mult.get(matched, 1.0) if matched else 1.0
        param_groups.append(
            {
                "params": params,
                "weight_decay": 0.0 if is_no_decay else wd,
                "lr_mult": mult,
            }
        )

    name = config.optimizer.name
    if name == "adamw":
        optimizer = AdamWOptimizer(
            param_groups,
            lr=config.optimizer.lr,
            betas=tuple(config.optimizer.betas),
            eps=config.optimizer.eps,
            fused=True,
        )
    elif name == "lion":
        # Lion has no `eps`; the field in OptimizerConfig is ignored.
        optimizer = LionOptimizer(
            param_groups,
            lr=config.optimizer.lr,
            betas=tuple(config.optimizer.betas),
        )
    else:
        raise ValueError(f"unknown optimizer: {name!r}; expected 'adamw' or 'lion'")
    return optimizer


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(
        self, optimizer, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        if self.current_step >= self.max_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / (
            self.max_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

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


class ConstantWarmupScheduler:
    """Linear warmup from 0 to max_lr over `warmup_steps`, then constant at max_lr."""

    def __init__(self, optimizer, warmup_steps: int, max_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)

    def get_lr(self):
        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        return self.max_lr

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.max_lr = state_dict.get("max_lr", self.max_lr)


def build_scheduler(optimizer, config: TrainConfig):
    """Build LR scheduler from config. Dispatches on config.scheduler.name."""
    name = config.scheduler.name
    if name == "cosine":
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr=config.scheduler.min_lr,
            max_lr=config.optimizer.lr,
        )
    if name == "constant":
        return ConstantWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            max_lr=config.optimizer.lr,
        )
    raise ValueError(f"unknown scheduler: {name!r}; expected 'cosine' or 'constant'")

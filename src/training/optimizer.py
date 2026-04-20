import math
import re
import torch
from src.utils.config import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
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
        param_groups.append({
            "params": params,
            "weight_decay": 0.0 if is_no_decay else wd,
            "lr_mult": mult,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        fused=True,
    )
    return optimizer


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float):
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
        progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

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


def build_scheduler(optimizer, config: TrainConfig):
    """Build LR scheduler from config."""
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.scheduler.min_lr,
        max_lr=config.optimizer.lr,
    )

import math
import torch
from src.utils.config import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied only to non-bias, non-layernorm params."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
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
            pg["lr"] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        if self.current_step >= self.max_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


def build_scheduler(optimizer, config: TrainConfig):
    """Build LR scheduler from config."""
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.scheduler.min_lr,
        max_lr=config.optimizer.lr,
    )

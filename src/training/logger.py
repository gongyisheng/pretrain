import wandb
from src.utils.config import TrainConfig


class WandbLogger:
    """Thin wrapper around W&B for training metrics."""

    def __init__(self, config: TrainConfig, enabled: bool = True):
        self.enabled = enabled
        self._on_log_hooks: list = []
        if self.enabled:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name or None,
                group=config.logging.wandb_group or None,
                config=config.to_dict(),
            )

    def register_on_log_hook(self, hook):
        """Register a callback fired after each log: hook(step, metrics)."""
        self._on_log_hooks.append(hook)

    def log(self, metrics: dict, step: int):
        if self.enabled:
            wandb.log(metrics, step=step)
        for hook in self._on_log_hooks:
            hook(step, metrics)

    def log_text(self, key: str, text: str, step: int):
        if self.enabled:
            wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()

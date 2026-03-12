import wandb
from src.utils.config import TrainConfig


class WandbLogger:
    """Thin wrapper around W&B for training metrics."""

    def __init__(self, config: TrainConfig, enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name or None,
                config=config.to_dict(),
            )

    def log(self, metrics: dict, step: int):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: int):
        if self.enabled:
            wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()

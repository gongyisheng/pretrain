import os
import torch
from src.utils.config import SpikeConfig


class SpikeDebugger:
    """Tracks grad norm spikes during training and saves full checkpoints,
    keeping only the top-K spikes by grad norm."""

    def __init__(self, config: SpikeConfig, checkpoint_dir: str):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._spike_checkpoints = []  # list of (grad_norm, ckpt_path)

    def on_step(self, grad_norm, step: int, model, save_checkpoint_fn, clip_value: float = 1.0) -> bool:
        """Call after gradient clipping.

        Args:
            grad_norm: pre-clip scalar grad norm (Tensor or float); returned by clip_grad_norm_
            step: current training step
            model: the model (to read .grad from named_parameters, post-clip)
            save_checkpoint_fn: callable() -> str that saves a full checkpoint and returns its path
            clip_value: grad_clip threshold used during training; used to reconstruct pre-clip norms

        Returns:
            True if a spike was detected and saved.
        """
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        threshold = self.config.grad_norm_threshold

        if grad_norm_val <= threshold:
            return False

        max_n = self.config.max_checkpoints

        # Skip if we're full and this spike doesn't beat the current minimum
        if len(self._spike_checkpoints) >= max_n and \
                grad_norm_val <= min(s[0] for s in self._spike_checkpoints):
            return False

        scale = grad_norm_val / clip_value if grad_norm_val > clip_value else 1.0
        raw_grads = {
            name: param.grad.norm().item() * scale
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        ckpt_path = save_checkpoint_fn(extra={"raw_grads": raw_grads})

        print(f"[debug] grad norm spike {grad_norm_val:.4f} > {threshold} at step {step}")

        self._spike_checkpoints.append((grad_norm_val, ckpt_path))

        # Evict the lowest grad norm spike if over limit
        if len(self._spike_checkpoints) > max_n:
            self._spike_checkpoints.sort(key=lambda s: s[0])
            evicted = self._spike_checkpoints.pop(0)
            if evicted[1] and os.path.exists(evicted[1]):
                os.remove(evicted[1])
            print(f"[debug] evicted spike checkpoint with grad_norm={evicted[0]:.4f}")

        return True

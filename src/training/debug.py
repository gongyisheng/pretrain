import os
import torch
from src.utils.config import SpikeConfig


class SpikeDebugger:
    """Tracks grad norm spikes during training and saves per-param grad norms and
    optional full checkpoints, keeping only the top-K spikes by grad norm."""

    def __init__(self, config: SpikeConfig, checkpoint_dir: str):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._spike_checkpoints = []  # list of (grad_norm, spike_path, ckpt_path)

    def on_step(self, grad_norm, step: int, model, save_checkpoint_fn) -> bool:
        """Call after logging.

        Args:
            grad_norm: scalar grad norm (Tensor or float)
            step: current training step
            model: the model (to read .grad from named_parameters)
            save_checkpoint_fn: callable() -> str that saves a full checkpoint and returns its path

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

        # Save per-param grad norms
        spike_data = {
            name: param.grad.norm().item()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        spike_path = os.path.join(self.checkpoint_dir, f"grad_spike_step_{step}.pt")
        torch.save(spike_data, spike_path)

        # Optionally save a full checkpoint
        ckpt_path = None
        if self.config.save_checkpoint:
            ckpt_path = save_checkpoint_fn()

        print(f"[debug] grad norm spike {grad_norm_val:.4f} > {threshold} at step {step}")

        self._spike_checkpoints.append((grad_norm_val, spike_path, ckpt_path))

        # Evict the lowest grad norm spike if over limit
        if len(self._spike_checkpoints) > max_n:
            self._spike_checkpoints.sort(key=lambda s: s[0])
            evicted = self._spike_checkpoints.pop(0)
            for path in evicted[1:]:
                if path and os.path.exists(path):
                    os.remove(path)
            print(f"[debug] evicted spike checkpoint with grad_norm={evicted[0]:.4f}")

        return True

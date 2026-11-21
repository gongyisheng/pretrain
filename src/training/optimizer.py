import math
import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.utils.config import TrainConfig


AdamWOptimizer = torch.optim.AdamW


class AdamCOptimizer(torch.optim.AdamW):
    """AdamC with corrected decay for selected groups. https://arxiv.org/abs/2506.02285"""

    @torch.no_grad()
    def step(self, closure=None):
        original_weight_decays = []
        try:
            for group in self.param_groups:
                if not group.get("use_corrected_weight_decay", False):
                    continue
                if group["lr"] == 0.0:
                    continue
                original_weight_decays.append((group, group["weight_decay"]))
                group["weight_decay"] *= group["lr"] / group["max_lr"]
            return super().step(closure)
        finally:
            for group, weight_decay in original_weight_decays:
                group["weight_decay"] = weight_decay


def _hyperball_step(
    param: torch.Tensor,
    direction: torch.Tensor,
    radius: torch.Tensor,
    lr: float,
    eps: float,
) -> None:
    direction = direction.to(dtype=radius.dtype)
    direction_norm = direction.norm(dim=(-2, -1), keepdim=True)
    direction = direction / direction_norm.clamp_min(eps)
    trial = param.to(dtype=radius.dtype) - lr * radius * direction
    trial_norm = trial.norm(dim=(-2, -1), keepdim=True)
    projected = radius * trial / trial_norm.clamp_min(eps)
    param.copy_(torch.where((direction_norm > 0) & (trial_norm > 0), projected, param))


class AdamHOptimizer(torch.optim.Adam):
    """Adam on fixed Frobenius spheres. https://arxiv.org/abs/2606.16899"""

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        param_names = param_group.get("param_names")
        if all(
            _is_hyperball_param(
                param_names[index] if param_names is not None else "", param
            )
            for index, param in enumerate(param_group["params"])
        ):
            param_group["weight_decay"] = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            param_names = group.get("param_names")
            for index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                if param.grad.layout != torch.strided:
                    raise RuntimeError("AdamH does not support sparse gradients")
                name = param_names[index] if param_names is not None else ""
                is_hyperball_param = _is_hyperball_param(name, param)
                # Keep moments and projection arithmetic at least float32.
                dtype = torch.promote_types(param.dtype, torch.float32)
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=dtype)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=dtype)
                if "radius" not in state and is_hyperball_param:
                    state["radius"] = torch.linalg.vector_norm(
                        param, dim=(-2, -1), keepdim=True, dtype=dtype
                    )
                state["step"] += 1
                grad = param.grad.to(dtype=dtype)
                exp_avg = state["exp_avg"].lerp_(grad, 1 - beta1)
                exp_avg_sq = state["exp_avg_sq"].mul_(beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)
                if lr == 0.0:
                    continue
                step = state["step"]
                bias_correction = 1 - beta1**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(1 - beta2**step)).add_(eps)
                if eps == 0.0:
                    denom.masked_fill_(denom == 0, 1.0)
                if not is_hyperball_param:
                    weight = param.to(dtype=dtype)
                    if weight_decay != 0.0:
                        weight.mul_(1 - lr * weight_decay)
                    weight.addcdiv_(exp_avg, denom, value=-lr / bias_correction)
                    if weight is not param:
                        param.copy_(weight)
                    continue

                direction = exp_avg.div(denom).div_(bias_correction)
                _hyperball_step(
                    param, direction, state["radius"], lr, torch.finfo(dtype).tiny
                )
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Restore the saved step counter and unrounded state tensors.
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
            for param, saved_id in zip(group["params"], saved_group["params"]):
                saved_state = state_dict["state"].get(saved_id, {})
                if "step" in saved_state:
                    self.state[param]["step"] = int(saved_state["step"])
                dtype = torch.promote_types(param.dtype, torch.float32)
                if dtype == param.dtype:
                    continue
                for key in ("exp_avg", "exp_avg_sq", "radius"):
                    if key in saved_state:
                        self.state[param][key] = saved_state[key].to(
                            device=param.device, dtype=dtype, copy=True
                        )


ADAM_OPTIMIZER_REGISTRY = {
    "adamw": AdamWOptimizer,
    "adamc": AdamCOptimizer,
    "adamh": AdamHOptimizer,
}


class LionOptimizer(torch.optim.Optimizer):
    """Lion sign-momentum optimizer. https://arxiv.org/abs/2302.06675"""

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
    def _single_tensor_lion(params, grads, exp_avgs, lr, beta1, beta2, wd):
        for p, g, m in zip(params, grads, exp_avgs):
            update = m.mul(beta1).add_(g, alpha=1 - beta1).sign_()
            p.mul_(1 - lr * wd).add_(update, alpha=-lr)
            m.mul_(beta2).add_(g, alpha=1 - beta2)

    @staticmethod
    def _multi_tensor_lion(params, grads, exp_avgs, lr, beta1, beta2, wd):
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


def _is_muon_param(name: str, param: torch.nn.Parameter) -> bool:
    """Select matrices other than embeddings, the output head, and routers."""
    return (
        param.ndim >= 2
        and "emb" not in name
        and "lm_head" not in name
        and "router" not in name
    )


def _is_hyperball_param(name: str, param: torch.nn.Parameter) -> bool:
    return _is_muon_param(name, param) and "ln" not in name and "bias" not in name


def _is_muons_param(param: torch.nn.Parameter) -> bool:
    """Select every matrix parameter."""
    return param.ndim >= 2


def _is_corrected_weight_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    """Select matrix weights other than embeddings and the output head."""
    return param.ndim >= 2 and "emb" not in name and "lm_head" not in name


def _adjust_lr(lr: float, adjust_lr_fn: str | None, dim_out: int, dim_in: int) -> float:
    """Scale Muon's learning rate by matrix shape."""
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, dim_out / dim_in))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(dim_out, dim_in))
    else:
        ratio = 1.0
    return lr * ratio


def _ns_iterate(X, a, b, c, ns_steps, eps):
    """Run quintic Newton-Schulz on a batch of matrices."""
    transposed = X.size(-2) > X.size(-1)  # work on the smaller Gram side
    if transposed:
        X = X.mT
    # Spectral norm <= 1 via per-matrix Frobenius norm.
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(ns_steps):
        gram = X @ X.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        X = torch.baddbmm(X, gram_update, X, beta=a)
    return X.mT if transposed else X


def _newton_schulz(grad, ns_coefficients, max_batch_elems, ns_steps, eps):
    """Orthogonalize 2D or batched matrix gradients in bfloat16 chunks."""
    a, b, c = ns_coefficients
    dim_out, dim_in = grad.size(-2), grad.size(-1)
    leading = grad.shape[:-2]
    X = grad.bfloat16().reshape(-1, dim_out, dim_in)  # (B, out, in), B=1 if 2D
    max_batch = max(1, max_batch_elems // (dim_out * dim_in))
    if X.size(0) > max_batch:
        X = torch.cat(
            [
                _ns_iterate(X[i : i + max_batch], a, b, c, ns_steps, eps)
                for i in range(0, X.size(0), max_batch)
            ]
        )
    else:
        X = _ns_iterate(X, a, b, c, ns_steps, eps)
    return X.reshape(*leading, dim_out, dim_in)


class MuonOptimizer(torch.optim.Optimizer):
    """Muon for 2D and batched MoE matrices. https://kellerjordan.github.io/posts/muon/"""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        ns_max_batch_elems: int = 8_000_000,
        eps: float = 1e-8,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=tuple(ns_coefficients),
            ns_max_batch_elems=ns_max_batch_elems,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon needs matrix params (ndim>=2), got shape {tuple(p.size())}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            buckets: dict = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                buckets.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)

            for (shape, _, _), params in buckets.items():
                grads = [p.grad for p in params]
                bufs = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    bufs.append(state["momentum_buffer"])

                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                updates = (
                    torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs
                )

                ortho = _newton_schulz(
                    torch.stack(updates),  # (K, *shape)
                    group["ns_coefficients"],
                    group["ns_max_batch_elems"],
                    group["ns_steps"],
                    group["eps"],
                )

                adjusted_lr = _adjust_lr(
                    lr, group["adjust_lr_fn"], shape[-2], shape[-1]
                )
                torch._foreach_mul_(params, 1 - lr * wd)
                torch._foreach_add_(params, list(ortho.unbind(0)), alpha=-adjusted_lr)
        return loss


class MuonCOptimizer(MuonOptimizer):
    """Muon with corrected weight decay. https://arxiv.org/abs/2506.02285"""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        ns_max_batch_elems: int = 8_000_000,
        eps: float = 1e-8,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ):
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            ns_max_batch_elems=ns_max_batch_elems,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        for group in self.param_groups:
            group.setdefault("use_corrected_weight_decay", True)
            group.setdefault("max_lr", group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        original_weight_decays = []
        try:
            for group in self.param_groups:
                if not group.get("use_corrected_weight_decay", False):
                    continue
                if group["lr"] == 0.0:
                    continue
                original_weight_decays.append((group, group["weight_decay"]))
                group["weight_decay"] *= group["lr"] / group["max_lr"]
            return super().step(closure)
        finally:
            for group, weight_decay in original_weight_decays:
                group["weight_decay"] = weight_decay


class MuonHOptimizer(MuonOptimizer):
    """Muon on fixed Frobenius spheres. https://arxiv.org/abs/2606.16899"""

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        param_names = param_group.get("param_names")
        if all(
            _is_hyperball_param(
                param_names[index] if param_names is not None else "", param
            )
            for index, param in enumerate(param_group["params"])
        ):
            param_group.update(weight_decay=0.0, adjust_lr_fn=None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            buckets: dict = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                buckets.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)

            for params in buckets.values():
                grads = [p.grad for p in params]
                momentum_buffers = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    if "radius" not in state:
                        state["radius"] = p.float().norm(dim=(-2, -1), keepdim=True)
                    momentum_buffers.append(state["momentum_buffer"])

                torch._foreach_lerp_(momentum_buffers, grads, 1 - momentum)
                updates = (
                    torch._foreach_lerp(grads, momentum_buffers, momentum)
                    if nesterov
                    else momentum_buffers
                )

                if lr == 0.0:
                    continue

                ortho = _newton_schulz(
                    torch.stack(updates),
                    group["ns_coefficients"],
                    group["ns_max_batch_elems"],
                    group["ns_steps"],
                    group["eps"],
                )

                for p, direction in zip(params, ortho.unbind(0)):
                    _hyperball_step(
                        p, direction, self.state[p]["radius"], lr, group["eps"]
                    )
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # The base loader casts floating state to the parameter dtype.
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
            for param, saved_id in zip(group["params"], saved_group["params"]):
                saved_state = state_dict["state"].get(saved_id, {})
                if "radius" in saved_state:
                    self.state[param]["radius"] = saved_state["radius"].to(
                        device=param.device, dtype=torch.float32, copy=True
                    )


MUON_OPTIMIZER_REGISTRY = {
    "muon": MuonOptimizer,
    "muonc": MuonCOptimizer,
    "muons": MuonOptimizer,
    "muonh": MuonHOptimizer,
}


class MuonAdamOptimizer:
    """Muon variants for selected matrices and an Adam variant for other weights.

    Muon: https://kellerjordan.github.io/posts/muon/
    MuonS: https://openreview.net/forum?id=ppmyFtr9EW
    """

    def __init__(
        self,
        muon_groups,
        adam_groups,
        lr: float,
        muon_cls: str = "muon",
        muon_kwargs: dict | None = None,
        adam_cls: str = "adamw",
        adam_kwargs: dict | None = None,
    ):
        muon_optimizer_cls = MUON_OPTIMIZER_REGISTRY.get(muon_cls)
        if muon_optimizer_cls is None:
            raise ValueError(f"unknown Muon optimizer: {muon_cls!r}")
        adam_optimizer_cls = ADAM_OPTIMIZER_REGISTRY.get(adam_cls)
        if adam_optimizer_cls is None:
            raise ValueError(f"unknown Adam companion: {adam_cls!r}")
        if muon_groups or adam_groups:
            muon_groups = muon_groups or [{"params": []}]
            adam_groups = adam_groups or [{"params": []}]
        self.muon_cls = muon_cls
        muon_kwargs = {} if muon_kwargs is None else muon_kwargs
        self.muon = muon_optimizer_cls(muon_groups, lr=lr, **muon_kwargs)
        self.adam_cls = adam_cls
        adam_kwargs = {} if adam_kwargs is None else adam_kwargs
        self.adam = adam_optimizer_cls(adam_groups, lr=lr, **adam_kwargs)
        self._optimizers = [self.muon, self.adam]

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    @property
    def state(self):
        merged = {}
        for opt in self._optimizers:
            merged.update(opt.state)
        return merged

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for opt in self._optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "muon_cls": self.muon_cls,
            "adam": self.adam.state_dict(),
            "adam_cls": self.adam_cls,
        }

    def load_state_dict(self, state_dict):
        saved_muon_cls = state_dict["muon_cls"]
        if saved_muon_cls != self.muon_cls:
            raise ValueError(
                f"checkpoint Muon optimizer {saved_muon_cls!r} does not match "
                f"configured optimizer {self.muon_cls!r}"
            )
        saved_adam_cls = state_dict["adam_cls"]
        if saved_adam_cls != self.adam_cls:
            raise ValueError(
                f"checkpoint Adam companion {saved_adam_cls!r} does not match "
                f"configured companion {self.adam_cls!r}"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])


OPTIMIZER_REGISTRY = {
    **ADAM_OPTIMIZER_REGISTRY,
    "lion": LionOptimizer,
    "muonadam": MuonAdamOptimizer,
}


def build_optimizer(
    model: torch.nn.Module, config: "TrainConfig"
) -> torch.optim.Optimizer:
    """Build optimizer groups with decay and per-parameter learning-rate multipliers.

    The first matching `lr_mult` regex sets each parameter's multiplier.
    """
    optimizer_name = config.optimizer.optimizer_cls
    kwargs = config.optimizer.optimizer_kwargs
    if optimizer_name == "muonadam":
        muon_name = kwargs["muon_cls"]
        muon_optimizer = MUON_OPTIMIZER_REGISTRY[muon_name]
        default_optimizer = ADAM_OPTIMIZER_REGISTRY[kwargs["adam_cls"]]
    else:
        muon_name = None
        muon_optimizer = None
        default_optimizer = OPTIMIZER_REGISTRY[optimizer_name]
    corrected_optimizers = (AdamCOptimizer, MuonCOptimizer)
    hyperball_optimizers = (AdamHOptimizer, MuonHOptimizer)
    patterns = [(re.compile(k), k) for k in config.optimizer.lr_mult.keys()]
    groups: dict = {}

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = None
        for rx, key in patterns:
            if rx.search(param_name):
                matched = key
                break
        is_no_decay = param.ndim < 2 or "ln" in param_name or "bias" in param_name
        if muon_optimizer is MuonHOptimizer:
            use_muon = _is_hyperball_param(param_name, param)
        elif muon_name == "muons":
            use_muon = _is_muons_param(param)
        else:
            use_muon = muon_optimizer is not None and _is_muon_param(param_name, param)
        param_optimizer = muon_optimizer if use_muon else default_optimizer
        use_corrected_weight_decay = (
            param_optimizer in corrected_optimizers
            and not is_no_decay
            and _is_corrected_weight_decay_param(param_name, param)
        )
        group_params = groups.setdefault(
            (matched, is_no_decay, param_optimizer, use_corrected_weight_decay),
            [],
        )
        group_params.append(
            (param_name, param) if param_optimizer is AdamHOptimizer else param
        )

    muon_groups = []
    param_groups = []
    for (
        matched,
        is_no_decay,
        group_optimizer,
        use_corrected_weight_decay,
    ), params in groups.items():
        mult = config.optimizer.lr_mult.get(matched, 1.0) if matched else 1.0
        group = {
            "params": params,
            "lr_mult": mult,
            "use_corrected_weight_decay": use_corrected_weight_decay,
        }
        if is_no_decay:
            group["weight_decay"] = 0.0
        if group_optimizer in (*corrected_optimizers, *hyperball_optimizers):
            group["lr"] = config.optimizer.lr * mult
        if group_optimizer in corrected_optimizers:
            group["max_lr"] = group["lr"]
        if group_optimizer is muon_optimizer:
            muon_groups.append(group)
        else:
            param_groups.append(group)

    if optimizer_name == "muonadam":
        return MuonAdamOptimizer(
            muon_groups,
            param_groups,
            lr=config.optimizer.lr,
            muon_cls=kwargs["muon_cls"],
            muon_kwargs=kwargs["muon_kwargs"],
            adam_cls=kwargs["adam_cls"],
            adam_kwargs=kwargs["adam_kwargs"],
        )
    return default_optimizer(param_groups, lr=config.optimizer.lr, **kwargs)


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


SCHEDULER_REGISTRY = {
    "cosine": CosineWarmupScheduler,
    "constant": ConstantWarmupScheduler,
}


def build_scheduler(optimizer, config: "TrainConfig"):
    """Build the configured learning-rate scheduler."""
    name = config.scheduler.name
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"unknown scheduler: {name!r}; expected one of {sorted(SCHEDULER_REGISTRY)}"
        )
    if name == "cosine":
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr=config.scheduler.min_lr,
            max_lr=config.optimizer.lr,
        )
    return ConstantWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        max_lr=config.optimizer.lr,
    )

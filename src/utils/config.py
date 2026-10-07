import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml

from src.layers.activation import ACT_REGISTRY
from src.layers.attention import ATTN_REGISTRY
from src.layers.mlp import MLP_REGISTRY, MOE_ROUTER_SCORE_FNS
from src.layers.pos_emb import POS_EMB_REGISTRY
from src.quant.constants import (
    QUANT_FORMATS,
    QUANT_GRANULARITY,
    QUANT_PASSTHROUGH,
    GEMM_OPS_BY_TENSOR,
    GEMM_TENSORS,
    QUANT_ROUNDING,
)
from src.quant.rotation import ROTATION_REGISTRY
from src.training.loss import LOSS_REGISTRY
from src.training.optimizer import (
    ADAM_OPTIMIZER_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
)

_MIXED_PRECISION = frozenset({"no", "bf16", "fp16"})
_DEVICES = frozenset({"auto", "cuda", "cpu"})
_SCALE_DTYPES = frozenset({"fp32", "fp8_e8m0", "fp8_e4m3"})


def _check_value(label, value, options) -> None:
    """Raise unless value names one of options."""
    try:
        known = value in options
    except TypeError:  # unhashable values never name a choice
        known = False
    if not known:
        raise ValueError(
            f"unknown {label}: {value!r}; expected one of {sorted(options)}"
        )


def _check_type(label, value, expected_type) -> None:
    """Raise unless value has the expected type."""
    if not isinstance(value, expected_type):
        raise ValueError(f"{label} must be a {expected_type.__name__}, got {value!r}")


def _check_include_exclude(label, include, exclude) -> None:
    """Raise unless include and exclude are lists of strings."""
    for selector, patterns in (("include", include), ("exclude", exclude)):
        _check_type(f"{label} {selector}", patterns, list)
        for pattern in patterns:
            _check_type(f"{label} {selector} pattern", pattern, str)


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 50257

    attn: list = field(default_factory=lambda: [{"attn_cls": "gqa", "attn_kwargs": {}}])
    mlp: list = field(default_factory=lambda: [{"mlp_cls": "dense", "mlp_kwargs": {}}])
    norm_cls: str = "rmsnorm"
    norm_kwargs: dict = field(default_factory=dict)
    pos_emb_cls: str = "rope"
    pos_emb_kwargs: dict = field(default_factory=dict)
    residual_cls: str = "standard"
    residual_kwargs: dict = field(default_factory=dict)

    dropout_embd: float = 0.0
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    def __post_init__(self):
        self._post_init_attn()
        self._post_init_mlp()

    def _set_default_attn_kwargs(self, attn_cls: str, kwargs: dict) -> None:
        """Fill defaults/validation for one attn item's kwargs, keyed on attn_cls."""
        kwargs.setdefault("attn_implementation", "flex_attention")
        if kwargs["attn_implementation"] == "sdpa" and kwargs.get("attn_sink", False):
            raise ValueError("attn_sink requires flex_attention implementation")
        n_heads = kwargs.get("n_heads")
        if n_heads is not None:
            kwargs.setdefault("attn_sink", False)
            # MLA sets its head dims explicitly, so d_model need not divide n_heads.
            if attn_cls in ("mha", "gqa") and self.d_model % n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})"
                )
            if attn_cls == "gqa":
                n_kv = kwargs.setdefault("n_kv_heads", n_heads)
                if n_heads % n_kv != 0:
                    raise ValueError(
                        f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv})"
                    )
            elif attn_cls == "mla":
                # Decoupled-RoPE head dims default off d_model // n_heads;
                # q_lora_rank=0 disables query compression.
                head_dim = self.d_model // n_heads
                kwargs.setdefault("qk_nope_head_dim", head_dim)
                kwargs.setdefault("qk_rope_head_dim", max(head_dim // 2, 1))
                kwargs.setdefault("v_head_dim", head_dim)
                kwargs.setdefault("kv_lora_rank", 4 * head_dim)
                kwargs.setdefault("q_lora_rank", 0)

    def _post_init_attn(self) -> None:
        if not self.attn:
            raise ValueError("model.attn must have at least one item")
        for item in self.attn:
            if "attn_cls" not in item:
                raise ValueError("each model.attn item requires 'attn_cls'")
            _check_value("attn_cls", item["attn_cls"], ATTN_REGISTRY)
            item.setdefault("attn_kwargs", {})
            self._set_default_attn_kwargs(item["attn_cls"], item["attn_kwargs"])

        n = self.n_layers
        claimed: dict[int, int] = {}
        bare: list[int] = []
        for idx, item in enumerate(self.attn):
            layer_idx = item.get("layer_idx")
            if layer_idx is None:
                bare.append(idx)
                continue
            seen = set()
            for layer in layer_idx:
                if layer in seen:
                    raise ValueError(
                        f"layer {layer} listed more than once in one attn item's layer_idx"
                    )
                seen.add(layer)
                if not (0 <= layer < n):
                    raise ValueError(f"layer_idx {layer} out of range [0, {n})")
                if layer in claimed:
                    raise ValueError(f"layer {layer} claimed by multiple attn items")
                claimed[layer] = idx
        if len(bare) > 1:
            raise ValueError("at most one model.attn item may omit layer_idx")
        if bare:
            remaining = [layer for layer in range(n) if layer not in claimed]
            self.attn[bare[0]]["layer_idx"] = remaining
            for layer in remaining:
                claimed[layer] = bare[0]
        missing = [layer for layer in range(n) if layer not in claimed]
        if missing:
            raise ValueError(
                f"layers {missing} have no attn item; add a fallback item without layer_idx"
            )

        self._layer_attn = [
            (
                self.attn[claimed[layer]]["attn_cls"],
                self.attn[claimed[layer]]["attn_kwargs"],
            )
            for layer in range(n)
        ]

        impls = {kw["attn_implementation"] for _, kw in self._layer_attn}
        if len(impls) > 1:
            raise ValueError(
                "all attn layers must share the same attn_implementation (the "
                f"trainer builds one attention mask shared across layers); got {sorted(impls)}"
            )

        if POS_EMB_REGISTRY[self.pos_emb_cls].rotary:
            dims = set()
            for attn_cls, attn_kwargs in self._layer_attn:
                if attn_cls == "mla":
                    dim = attn_kwargs.get("qk_rope_head_dim")
                elif attn_kwargs.get("n_heads") is not None:
                    dim = self.d_model // attn_kwargs["n_heads"]
                else:
                    dim = None
                if dim is not None:
                    dims.add(dim)
            if len(dims) > 1:
                raise ValueError(
                    "rotary pos_emb requires a single rope head-dim across layers; "
                    f"got {sorted(dims)}. Make qk_rope_head_dim / (d_model // "
                    "n_heads) match."
                )

    def resolve_attn(self, layer_idx: int) -> tuple[str, dict]:
        return self._layer_attn[layer_idx]

    @property
    def attn_implementation(self) -> str:
        """Shared attn_implementation across all layers (validated in _post_init_attn)."""
        return self._layer_attn[0][1]["attn_implementation"]

    def _set_default_mlp_kwargs(self, mlp_cls: str, kwargs: dict) -> None:
        """Fill defaults/validation for one MLP item's kwargs, keyed on mlp_cls."""
        kwargs.setdefault("intermediate_size", 4 * self.d_model)
        activation_cls = kwargs.setdefault("activation_cls", "swiglu")
        _check_value("activation_cls", activation_cls, ACT_REGISTRY)
        activation_kwargs = kwargs.setdefault("activation_kwargs", {})
        if not isinstance(activation_kwargs, dict):
            del kwargs["activation_kwargs"]
            activation_kwargs = {}
        act_limit = activation_kwargs.get("act_limit")
        if act_limit is not None and not isinstance(act_limit, dict):
            del activation_kwargs["act_limit"]
            act_limit = None
        if act_limit is not None:
            for side in ("gate", "up"):
                bounds = act_limit.get(side)
                if not isinstance(bounds, dict):
                    continue
                lo, hi = bounds.get("min"), bounds.get("max")
                if not isinstance(lo, (int, float)):
                    lo = None
                if not isinstance(hi, (int, float)):
                    hi = None
                if lo is not None and hi is not None and lo >= hi:
                    raise ValueError(
                        f"act_limit['{side}'] requires min < max; "
                        f"got min={lo!r}, max={hi!r}"
                    )
        if mlp_cls == "moe":
            kwargs.setdefault("n_shared_experts", 0)
            kwargs.setdefault("bias", False)
            kwargs.setdefault("router_score_fn", "sigmoid")
            _check_value(
                "router_score_fn", kwargs["router_score_fn"], MOE_ROUTER_SCORE_FNS
            )
            # aux_loss (Switch) and expert_bias (arXiv:2408.15664) are mutually
            # exclusive balancing strategies; exactly one must be enabled.
            kwargs.setdefault("expert_bias", False)
            kwargs.setdefault("aux_loss", False)
            if not kwargs["aux_loss"] and not kwargs["expert_bias"]:
                raise ValueError(
                    "exactly one of aux_loss / expert_bias must be enabled; both are off"
                )
            if kwargs["aux_loss"] and kwargs["expert_bias"]:
                raise ValueError(
                    "aux_loss and expert_bias are mutually exclusive; both are on"
                )
            if kwargs["expert_bias"]:
                kwargs.setdefault("expert_bias_update_rate", 0.001)
            if kwargs["aux_loss"]:
                kwargs.setdefault("aux_loss_coef", 0.001)
            kwargs.setdefault("latent_moe", False)
            if kwargs["latent_moe"]:
                latent_dim = kwargs.get("latent_dim")
                if (
                    not isinstance(latent_dim, int)
                    or isinstance(latent_dim, bool)
                    or latent_dim <= 0
                ):
                    raise ValueError(
                        f"latent_dim must be a positive int when latent_moe=True; "
                        f"got {latent_dim!r}"
                    )

    def _post_init_mlp(self) -> None:
        if not self.mlp:
            raise ValueError("model.mlp must have at least one item")
        for item in self.mlp:
            if "mlp_cls" not in item:
                raise ValueError("each model.mlp item requires 'mlp_cls'")
            _check_value("mlp_cls", item["mlp_cls"], MLP_REGISTRY)
            item.setdefault("mlp_kwargs", {})
            self._set_default_mlp_kwargs(item["mlp_cls"], item["mlp_kwargs"])

        n = self.n_layers
        claimed: dict[int, int] = {}  # layer -> item index
        bare: list[int] = []
        for idx, item in enumerate(self.mlp):
            layer_idx = item.get("layer_idx")
            if layer_idx is None:
                bare.append(idx)
                continue
            if len(set(layer_idx)) != len(layer_idx):
                dupe = next(layer for layer in layer_idx if layer_idx.count(layer) > 1)
                raise ValueError(
                    f"layer {dupe} listed more than once in one mlp item's layer_idx"
                )
            for layer in layer_idx:
                if not (0 <= layer < n):
                    raise ValueError(f"layer_idx {layer} out of range [0, {n})")
                if layer in claimed:
                    raise ValueError(f"layer {layer} claimed by multiple mlp items")
                claimed[layer] = idx
        if len(bare) > 1:
            raise ValueError("at most one model.mlp item may omit layer_idx")
        if bare:
            remaining = [layer for layer in range(n) if layer not in claimed]
            self.mlp[bare[0]]["layer_idx"] = remaining
            for layer in remaining:
                claimed[layer] = bare[0]
        missing = [layer for layer in range(n) if layer not in claimed]
        if missing:
            raise ValueError(
                f"layers {missing} have no mlp item; add a fallback item without layer_idx"
            )

        # per-layer (cls, kwargs) map; plain attribute, not a dataclass field,
        # so asdict()/to_dict() serialize only the `mlp` list.
        self._layer_mlp = [
            (
                self.mlp[claimed[layer]]["mlp_cls"],
                self.mlp[claimed[layer]]["mlp_kwargs"],
            )
            for layer in range(n)
        ]

    def resolve_mlp(self, layer_idx: int) -> tuple[str, dict]:
        return self._layer_mlp[layer_idx]

    @property
    def is_moe(self) -> bool:
        return any(cls == "moe" for cls, _ in self._layer_mlp)


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4
    prefetch_factor: int = 4
    packing: bool = True
    tokenizer_path: str = "tokenizers/custom_bpe"


@dataclass
class TokenizerTrainingConfig:
    method: str = "bpe"  # "bpe" | "superbpe"
    method_kwargs: dict = field(default_factory=dict)
    num_samples: int = 1_000_000
    checkpoint_dir: str = "tokenizers/custom_bpe"
    checkpoint_every: int = 5000
    eval_every: int = 5000

    def __post_init__(self):
        self.method_kwargs.setdefault("eval_num_docs", 1000)
        if self.method == "superbpe":
            self.method_kwargs.setdefault("max_superword_words", 4)


@dataclass
class QuantizationConfig:
    enabled: bool = False
    # {tensor: fmt} or {tensor: {gemm: fmt}}, resolved to the latter by __post_init__
    dtype: dict = field(default_factory=dict)
    # {tensor: {granularity, block_shape}, scale_dtype, enable_global_scale}
    scale: dict = field(default_factory=dict)
    rounding: dict = field(default_factory=dict)  # {tensor: "RNE" | "SR"}
    rotation: dict = field(default_factory=dict)
    layer_idx: Optional[List[int]] = None
    enabled_after_steps: int = 0
    include: List[str] = field(default_factory=lambda: ["*"])
    exclude: List[str] = field(default_factory=lambda: ["lm_head", "*mlp.router.gate"])

    def __post_init__(self):
        _check_type("quant enabled", self.enabled, bool)
        _check_type("quant scale", self.scale, dict)
        _check_type("quant rounding", self.rounding, dict)
        _check_type("quant rotation", self.rotation, dict)
        self.rotation.setdefault("rotation_cls", None)
        self.rotation.setdefault("rotation_kwargs", {})
        self.rotation.setdefault("rotation_axes", {})
        if not self.enabled:
            return
        _check_include_exclude("quant", self.include, self.exclude)
        if (
            not isinstance(self.enabled_after_steps, int)
            or isinstance(self.enabled_after_steps, bool)
            or self.enabled_after_steps < 0
        ):
            raise ValueError("quant enabled_after_steps must be a nonnegative integer")
        if self.layer_idx is not None:
            self.layer_idx = list(
                dict.fromkeys(layer for layer in self.layer_idx if layer >= 0)
            )
        self._post_init_dtype()
        self._post_init_scale()
        self._post_init_rounding()
        self._post_init_rotation()

    def _post_init_dtype(self):
        """Expand {tensor: fmt} to {tensor: {gemm: fmt}}."""
        for tensor, value in self.dtype.items():
            _check_value("quant dtype key", tensor, GEMM_OPS_BY_TENSOR)
            gemms = GEMM_OPS_BY_TENSOR[tensor]
            per_gemm = (
                dict(value) if isinstance(value, dict) else dict.fromkeys(gemms, value)
            )
            for gemm, fmt in per_gemm.items():
                if gemm not in gemms:
                    raise ValueError(
                        f"quant dtype {tensor!r} cannot be scoped to {gemm!r}: "
                        f"only {list(gemms)} consume it"
                    )
                _check_value(f"quant fmt for {tensor}.{gemm}", fmt, QUANT_FORMATS)
            self.dtype[tensor] = per_gemm

    def _post_init_scale(self):
        """Resolve each tensor's scale layout."""
        scale_dtype = self.scale.get("scale_dtype")
        if scale_dtype is None:
            scale_dtype = "fp32"
        _check_value("quant scale_dtype", scale_dtype, _SCALE_DTYPES)

        resolved_scale = {}
        for tensor in GEMM_TENSORS:
            scale = self.scale.get(tensor, {})
            _check_type(f"quant scale.{tensor}", scale, dict)
            tensor_scale = {
                "granularity": scale.get("granularity", "tensorwise"),
                "block_shape": scale.get("block_shape"),
            }
            granularity = tensor_scale["granularity"]
            _check_value(
                f"quant granularity for {tensor}", granularity, QUANT_GRANULARITY
            )
            if granularity != "blockwise":
                tensor_scale["block_shape"] = (
                    (0, 0) if granularity == "tensorwise" else (1, 0)
                )
            else:
                shape = tensor_scale["block_shape"]
                if (
                    not isinstance(shape, (list, tuple))
                    or len(shape) != 2
                    or not all(isinstance(value, int) for value in shape)
                ):
                    raise ValueError(
                        f"quant scale.{tensor}.block_shape must be a pair of ints, "
                        f"got {shape!r}"
                    )
                outer, contract = shape
                if outer != 1 and outer != contract:
                    raise ValueError(
                        "quant block_shape must be 1D (1, N) or a square tile "
                        f"(N, N), got {shape!r}"
                    )
                if contract <= 0 or contract % 16 != 0:
                    raise ValueError(
                        "quant block_shape contract extent must be a positive "
                        f"multiple of 16, got {contract}"
                    )
                if scale_dtype == "fp8_e8m0" and contract % 32 != 0:
                    raise ValueError(
                        "quant scale_dtype 'fp8_e8m0' needs a contract extent that "
                        f"is a multiple of 32, the mx scale vector, got {contract}"
                    )
                tensor_scale["block_shape"] = (outer, contract)
            resolved_scale[tensor] = tensor_scale

        if scale_dtype == "fp8_e8m0":
            non_blockwise = [
                tensor
                for tensor, tensor_scale in resolved_scale.items()
                if tensor_scale["granularity"] != "blockwise"
            ]
            if non_blockwise:
                raise ValueError(
                    "quant scale_dtype 'fp8_e8m0' requires granularity 'blockwise' "
                    f"for {non_blockwise}"
                )

        # The e8m0 shared exponent only has fp8 kernels, so mxfp8 + int8 (or any
        # other non-fp8 element) is rejected here. Pass-through formats are exempt:
        # they are unquantized and carry no scale.
        if scale_dtype == "fp8_e8m0":
            for tensor, per_gemm in self.dtype.items():
                for gemm, fmt in per_gemm.items():
                    if fmt not in QUANT_PASSTHROUGH and not fmt.startswith("fp8"):
                        raise ValueError(
                            f"mxfp8 scale requires an fp8 element for "
                            f"{tensor}.{gemm}, got {fmt!r}"
                        )

        enable_global_scale = self.scale.get("enable_global_scale", False)
        _check_type("quant scale 'enable_global_scale'", enable_global_scale, bool)
        # Full arithmetic scale dtypes absorb a global factor. E8M0's exponent
        # range makes normalization overflow fp32, so neither needs one.
        if enable_global_scale and (
            scale_dtype in QUANT_PASSTHROUGH or scale_dtype == "fp8_e8m0"
        ):
            print(f"quant: disabled enable_global_scale for {scale_dtype!r} scales")
            enable_global_scale = False

        self.scale = {
            **resolved_scale,
            "scale_dtype": scale_dtype,
            "enable_global_scale": enable_global_scale,
        }

    def _post_init_rounding(self):
        """Validate the named rounding modes and default the rest to RNE."""
        for tensor, mode in self.rounding.items():
            _check_value("quant rounding key", tensor, GEMM_OPS_BY_TENSOR)
            _check_value(f"quant rounding for {tensor}", mode, QUANT_ROUNDING)
        self.rounding = {t: self.rounding.get(t, "RNE") for t in GEMM_TENSORS}

    def _post_init_rotation(self):
        """Canonicalize rotation to per-tensor, per-GEMM matrix axes."""
        rotation_cls = self.rotation["rotation_cls"]
        resolved_axes = {
            tensor: {gemm: [] for gemm in gemms}
            for tensor, gemms in GEMM_OPS_BY_TENSOR.items()
        }
        if rotation_cls is None:
            self.rotation = {
                "rotation_cls": None,
                "rotation_kwargs": {},
                "rotation_axes": resolved_axes,
            }
            return
        _check_value("rotation_cls", rotation_cls, ROTATION_REGISTRY)

        rotation_kwargs = self.rotation["rotation_kwargs"]
        _check_type("quant rotation 'rotation_kwargs'", rotation_kwargs, dict)
        rotation_kwargs = dict(rotation_kwargs)
        # build_rotation_key serializes these to derive a stable identity, and the
        # config itself must survive the YAML round trip.
        try:
            json.dumps(rotation_kwargs, sort_keys=True)
        except TypeError as error:
            raise ValueError(
                f"quant rotation 'rotation_kwargs' must hold only YAML/JSON "
                f"values, got {rotation_kwargs!r}"
            ) from error

        rotation_axes = self.rotation["rotation_axes"]
        _check_type("quant rotation 'rotation_axes'", rotation_axes, dict)
        for tensor, gemm_axes in rotation_axes.items():
            _check_value("quant rotation tensor", tensor, GEMM_OPS_BY_TENSOR)
            _check_type(f"quant rotation '{tensor}'", gemm_axes, dict)
            for gemm, axes in gemm_axes.items():
                _check_value(
                    f"quant rotation GEMM for {tensor}",
                    gemm,
                    GEMM_OPS_BY_TENSOR[tensor],
                )
                _check_type(f"quant rotation axes for {tensor}.{gemm}", axes, list)
                for axis in axes:
                    _check_type(f"quant rotation axis for {tensor}.{gemm}", axis, int)
                    _check_value(
                        f"quant rotation axis for {tensor}.{gemm}", axis, (-2, -1)
                    )
                resolved_axes[tensor][gemm] = sorted(set(axes))

        self.rotation = {
            "rotation_cls": rotation_cls,
            "rotation_kwargs": rotation_kwargs,
            "rotation_axes": resolved_axes,
        }


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_steps: int = 50000
    early_stop: int = 0
    device: str = "auto"  # "auto" (cuda if available else cpu) | "cuda" | "cpu"
    mixed_precision: str = "bf16"
    loss_fn: str = "cross_entropy"
    label_smoothing: float = 0.0  # for CE loss only
    enable_torch_compile: bool = True
    use_deterministic_algo: bool = False
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 100
    eval_steps: int = 25
    eval_batch_size: int = 16
    eval_train: bool = False  # for SFT
    eval_generate: bool = False
    intra_doc_masking: bool = True

    def __post_init__(self):
        _check_value("device", self.device, _DEVICES)
        _check_value("mixed_precision", self.mixed_precision, _MIXED_PRECISION)
        _check_value("loss_fn", self.loss_fn, LOSS_REGISTRY)


@dataclass
class OptimizerConfig:
    optimizer_cls: str = "adamw"  # "adamw" | "adamc" | "lion" | "muonadam"
    lr: float = 5e-4
    lr_mult: Dict[str, float] = field(default_factory=lambda: {"lm_head": 1.0})
    weight_decay: float = 0.1
    optimizer_kwargs: dict = field(default_factory=dict)

    def _post_init_adam(self, kwargs: dict) -> None:
        # beta2=0.95 (not torch's 0.999) is the GPT-3/LLaMA setting.
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("eps", 1e-8)
        kwargs.setdefault("fused", True)

    def _post_init_lion(self, kwargs: dict) -> None:
        kwargs.setdefault("betas", (0.9, 0.99))  # Chen et al. 2023 default
        kwargs.setdefault("foreach", True)

    def _post_init_muon(self, kwargs: dict) -> None:
        kwargs.setdefault("momentum", 0.95)
        kwargs.setdefault("nesterov", True)
        # Rescale Muon's update RMS to AdamW's so both halves share one lr.
        kwargs.setdefault("adjust_lr_fn", "match_rms_adamw")
        kwargs.setdefault("eps", 1e-8)

    def __post_init__(self):
        _check_value("optimizer_cls", self.optimizer_cls, OPTIMIZER_REGISTRY)
        kwargs = self.optimizer_kwargs
        if self.optimizer_cls in ADAM_OPTIMIZER_REGISTRY:
            self._post_init_adam(kwargs)
        elif self.optimizer_cls == "lion":
            self._post_init_lion(kwargs)
        elif self.optimizer_cls == "muonadam":
            adam_cls = kwargs.setdefault("adam_cls", "adamw")
            _check_value("adam_cls", adam_cls, ADAM_OPTIMIZER_REGISTRY)
            adam_kwargs = kwargs.setdefault("adam_kwargs", {})
            muon_kwargs = kwargs.setdefault("muon_kwargs", {})
            if not isinstance(adam_kwargs, dict):
                adam_kwargs = {}
                kwargs["adam_kwargs"] = adam_kwargs
            if not isinstance(muon_kwargs, dict):
                muon_kwargs = {}
                kwargs["muon_kwargs"] = muon_kwargs
            self._post_init_adam(adam_kwargs)
            self._post_init_muon(muon_kwargs)


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 5e-5

    def __post_init__(self):
        _check_value("scheduler", self.name, SCHEDULER_REGISTRY)


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    wandb_group: str = ""
    log_every: int = 10
    log_grad_norms: bool = True
    log_weight_norms: bool = True
    log_activation_norms: bool = True
    log_grad_svd_metrics: bool = True
    log_weight_svd_metrics: bool = True
    log_optimizer_step_norms: bool = True
    log_quant_metrics: bool = True


@dataclass
class TrainConfig:
    task: str = "pretrain"  # "pretrain" | "sft"
    max_seq_len: int = 1024
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer_training: TokenizerTrainingConfig = field(
        default_factory=TokenizerTrainingConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        self._post_init_quantization_training_config()
        self._post_init_model_training_config()

    def _post_init_quantization_training_config(self):
        if self.quantization.enabled:
            amp_dtype = (
                "fp32"
                if self.training.mixed_precision == "no"
                else self.training.mixed_precision
            )
            for tensor, gemms in GEMM_OPS_BY_TENSOR.items():
                tensor_dtypes = self.quantization.dtype.setdefault(tensor, {})
                for gemm in gemms:
                    tensor_dtypes.setdefault(gemm, amp_dtype)

    def _post_init_model_training_config(self):
        m = self.model
        if m.is_moe and self.training.mixed_precision not in ("bf16", "fp16"):
            raise ValueError(
                "dropless MoE requires training.mixed_precision='bf16' or 'fp16'; "
                f"got {self.training.mixed_precision!r}."
            )

    def to_dict(self):
        return asdict(self)


def _apply_overrides(config: TrainConfig, overrides: List[str]):
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        field_name = parts[-1]
        current = (
            obj.get(field_name) if isinstance(obj, dict) else getattr(obj, field_name)
        )
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif current is None:
            # Optional field: try float, then int, then leave as string
            try:
                value = float(value)
            except ValueError:
                try:
                    value = int(value)
                except ValueError:
                    pass
        if isinstance(obj, dict):
            obj[field_name] = value
        else:
            setattr(obj, field_name, value)


def _coerce_types(dc_class, raw_dict: dict) -> dict:
    """Coerce raw YAML values to match dataclass field types.

    PyYAML safe_load treats scientific notation (e.g. 6e-4) as strings.
    This converts them to the correct type based on the dataclass annotation.
    """
    field_types = {f.name: f.type for f in dataclasses.fields(dc_class)}
    coerced = {}
    for k, v in raw_dict.items():
        if k not in field_types:
            continue  # silently ignore unknown/deprecated YAML fields
        expected = field_types.get(k)
        if expected is float and isinstance(v, str):
            v = float(v)
        elif expected is int and isinstance(v, str):
            v = int(v)
        elif expected is bool and isinstance(v, str):
            v = v.lower() in ("true", "1", "yes")
        coerced[k] = v
    return coerced


def _coerce_kwargs(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _coerce_kwargs(v)
        elif isinstance(v, str):
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass


def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig(
        task=raw.get("task", "pretrain"),
        max_seq_len=raw.get("max_seq_len", 1024),
        seed=raw.get("seed", 42),
        model=ModelConfig(**_coerce_types(ModelConfig, raw.get("model", {}))),
        data=DataConfig(**_coerce_types(DataConfig, raw.get("data", {}))),
        quantization=QuantizationConfig(
            **_coerce_types(QuantizationConfig, raw.get("quantization", {}))
        ),
        tokenizer_training=TokenizerTrainingConfig(
            **_coerce_types(TokenizerTrainingConfig, raw.get("tokenizer_training", {}))
        ),
        training=TrainingConfig(
            **_coerce_types(TrainingConfig, raw.get("training", {}))
        ),
        optimizer=OptimizerConfig(
            **_coerce_types(OptimizerConfig, raw.get("optimizer", {}))
        ),
        scheduler=SchedulerConfig(
            **_coerce_types(SchedulerConfig, raw.get("scheduler", {}))
        ),
        logging=LoggingConfig(**_coerce_types(LoggingConfig, raw.get("logging", {}))),
    )

    for kw in (
        config.model.norm_kwargs,
        config.model.pos_emb_kwargs,
        config.model.residual_kwargs,
        config.tokenizer_training.method_kwargs,
        config.optimizer.optimizer_kwargs,
    ):
        _coerce_kwargs(kw)
    for item in config.model.attn:
        _coerce_kwargs(item["attn_kwargs"])
    for item in config.model.mlp:
        _coerce_kwargs(item["mlp_kwargs"])
    if overrides:
        _apply_overrides(config, overrides)

    return config

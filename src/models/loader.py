"""
Loader for Draft Model (Qwen/Qwen2.5-0.5B) and Target Model (Qwen/Qwen2.5-7B).

Supports:
- In-memory singleton caching (`_MODEL_CACHE`) so repeated calls in a notebook,
  test suite, or server reuse already-loaded models in 0ms without reloading VRAM
- Local disk cache fast-path (`local_files_only=True` check before network calls)
  so once downloaded, weights load directly from disk without Hugging Face Hub pings
- Standard FP16 / BF16 / FP32 model loading
- Low-resource GPU 4-bit NF4 / 8-bit quantization via bitsandbytes
- Lightweight synthetic Qwen2 model pair creation for fast unit testing & CI
"""

from dataclasses import dataclass
import importlib.util
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from config import ModelConfig
from src.speculative.draft_attention import use_packed_gqa_draft_attention


@dataclass
class LoadedModels:
    """Container holding the draft model, target model, shared tokenizer, and device info."""

    draft_model: Optional[PreTrainedModel]
    target_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    attn_implementation: str = "sdpa"
    target_quant: str = "fp16"
    draft_attn_implementation: str = "sdpa"


# In-memory registries to prevent redundant model reloading within the same process/session
_MODEL_CACHE: Dict[Tuple[Any, ...], LoadedModels] = {}
_MOCK_MODEL_CACHE: Dict[Tuple[str, int], LoadedModels] = {}


def clear_model_cache() -> None:
    """Clears the in-memory model cache and frees GPU VRAM if needed."""
    _MODEL_CACHE.clear()
    _MOCK_MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device_and_dtype(config: ModelConfig) -> Tuple[torch.device, torch.dtype]:
    """Resolves the optimal execution device and floating-point precision."""
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    if device.type == "cpu":
        return device, torch.float32

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype.lower(), torch.float16)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return device, dtype


def is_flash_attn_available() -> bool:
    """Returns True when `flash_attn` is importable in the active Python environment."""
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


FLASH_ATTN_INSTALL_HINT = (
    "Install flash-attn in this runtime (e.g. in a Colab cell: "
    "`pip install flash-attn --no-build-isolation`), restart the runtime, and re-run."
)

_FA2_REQUESTS = ("auto", "flash_attention_2", "fa2")


class FlashAttentionUnavailableError(RuntimeError):
    """Raised when FlashAttention-2 is required but cannot be imported or loaded (no SDPA fallback)."""


def resolve_attn_implementation(
    requested: str,
    dtype: torch.dtype,
    device_type: str,
) -> str:
    """
    Resolves the attention backend. On CUDA, `"auto"` / `"flash_attention_2"` REQUIRE FlashAttention-2:
    if `flash_attn` is not importable (or dtype is not fp16/bf16) this raises
    `FlashAttentionUnavailableError` instead of falling back to SDPA.
    On CPU (mock models / unit tests) FA2 kernels cannot run, so `"auto"` resolves to `"sdpa"`;
    an explicit `"flash_attention_2"` request on CPU raises.
    Explicit `"sdpa"` / `"eager"` requests are returned unchanged.
    """
    req = (requested or "auto").lower()
    if req in _FA2_REQUESTS:
        if device_type != "cuda":
            if req == "auto":
                return "sdpa"
            raise FlashAttentionUnavailableError(
                f"attn_implementation='flash_attention_2' requires a CUDA GPU (device='{device_type}')."
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise FlashAttentionUnavailableError(
                f"FlashAttention-2 requires fp16/bf16 weights (got {dtype}). Run with --dtype float16."
            )
        if not is_flash_attn_available():
            raise FlashAttentionUnavailableError(
                "FlashAttention-2 is required but `flash_attn` is not importable. " + FLASH_ATTN_INSTALL_HINT
            )
        return "flash_attention_2"
    return req


def print_model_runtime_info(models: "LoadedModels") -> None:
    """Prints attn_implementation and dtype of both target and draft models (called at the start of every run)."""
    for role, model in (("Target", models.target_model), ("Draft", models.draft_model)):
        if model is None:
            print(f"[RunInfo] {role:<6}: not loaded")
            continue
        attn = getattr(model.config, "_attn_implementation", None)
        dtype = getattr(model, "dtype", None)
        name = getattr(model.config, "_name_or_path", "") or model.__class__.__name__
        print(f"[RunInfo] {role:<6}: attn_implementation={attn} | dtype={dtype} | model={name}")


def set_attn_implementation(models: LoadedModels, requested: str) -> str:
    """
    Dynamically switches the TARGET model to `requested` attention backend without reloading weights.
    The draft keeps the backend it was loaded with (SDPA + packed-GQA decode by default), because FA2 on the
    draft's StaticCache forces per-layer host syncs and blocks CUDA-graph capture.
    """
    resolved = resolve_attn_implementation(requested, models.dtype, models.device.device if hasattr(models.device, "device") else models.device.type)
    models.target_model.config._attn_implementation = resolved
    models.attn_implementation = resolved
    return resolved


def is_instruct_model_name(model_name: str) -> bool:
    return "instruct" in (model_name or "").lower()


def resolve_model_pair_for_quant(cfg: ModelConfig) -> Tuple[str, str, str]:
    """
    Step 3: Resolves effective `(target_quant, target_model_name, draft_model_name)`.
    Validates `target_quant` in {fp16, bnb4, awq} and prevents mixing Base & Instruct variants.
    """
    t_quant = (cfg.target_quant or "fp16").lower()
    if cfg.load_target_in_4bit and t_quant == "fp16":
        t_quant = "bnb4"
    if t_quant not in ("fp16", "bnb4", "awq"):
        raise ValueError(f"Unsupported target_quant '{cfg.target_quant}'. Choose from: fp16, bnb4, awq.")

    target_name = cfg.target_model_name
    draft_name = cfg.draft_model_name

    if t_quant == "awq":
        if target_name == "Qwen/Qwen2.5-7B":
            target_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"
            if draft_name == "Qwen/Qwen2.5-0.5B":
                draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
            print(
                "[ModelLoader] target_quant='awq': switching default Base pair to matched Instruct AWQ pair "
                f"({target_name} + {draft_name}, backend='{cfg.awq_backend}') so Base and Instruct are not mixed."
            )
        elif is_instruct_model_name(target_name) and draft_name == "Qwen/Qwen2.5-0.5B":
            draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
            print(
                f"[ModelLoader] target_quant='awq': auto-pairing Instruct target '{target_name}' "
                f"with '{draft_name}'."
            )

    if is_instruct_model_name(target_name) != is_instruct_model_name(draft_name):
        raise ValueError(
            f"Do not mix Base and Instruct models between target ('{target_name}') "
            f"and draft ('{draft_name}')."
        )

    return t_quant, target_name, draft_name


def check_vram_fit_for_sequence(
    target_model: PreTrainedModel,
    draft_model: Optional[PreTrainedModel],
    target_quant: str,
    seq_len: int = 16384,
) -> Dict[str, float]:
    """
    Step 3 / Step 6: Estimates model weight + KV cache footprint at `seq_len` against total GPU VRAM.
    """
    def _model_bytes(m: Optional[PreTrainedModel]) -> float:
        if m is None:
            return 0.0
        total = 0.0
        for p in m.parameters():
            total += float(p.numel() * p.element_size())
        return total

    def _kv_bytes_per_token(m: Optional[PreTrainedModel]) -> float:
        if m is None:
            return 0.0
        mc = m.config
        n_layers = int(getattr(mc, "num_hidden_layers", 28))
        n_kv = int(getattr(mc, "num_key_value_heads", getattr(mc, "num_attention_heads", 4)))
        head_dim = int(getattr(mc, "head_dim", getattr(mc, "hidden_size", 3584) // max(1, getattr(mc, "num_attention_heads", 28))))
        return float(n_layers * n_kv * head_dim * 2 * 2)  # K+V in FP16

    weights_gb = (_model_bytes(target_model) + _model_bytes(draft_model)) / (1024.0 ** 3)
    kv_gb = ((_kv_bytes_per_token(target_model) + _kv_bytes_per_token(draft_model)) * seq_len) / (1024.0 ** 3)
    est_total_gb = weights_gb + kv_gb + 1.25  # + ~1.25 GB CUDA workspace / activations

    gpu_total_gb = 0.0
    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
        status = "OK" if est_total_gb <= gpu_total_gb else "TIGHT/OOM-RISK"
        print(
            f"[VRAM Check @ {seq_len} tok | quant={target_quant}] "
            f"Weights={weights_gb:.2f}GB + KV({seq_len})={kv_gb:.2f}GB + WS~1.25GB "
            f"= {est_total_gb:.2f}GB / {gpu_total_gb:.2f}GB GPU [{status}]"
        )
    return {
        "weights_gb": round(weights_gb, 3),
        "kv_gb": round(kv_gb, 3),
        "estimated_total_gb": round(est_total_gb, 3),
        "gpu_total_gb": round(gpu_total_gb, 3),
        "fits": gpu_total_gb == 0.0 or est_total_gb <= gpu_total_gb,
    }


def _build_quantization_config(
    target_quant: str,
    load_in_8bit: bool,
    compute_dtype: torch.dtype,
    awq_backend: str = "marlin",
    model_name: str = "",
) -> Optional[Any]:
    if target_quant == "awq":
        try:
            from transformers import AwqConfig
            return AwqConfig(
                bits=4,
                group_size=128,
                zero_point=True,
                backend=awq_backend,
            )
        except Exception as exc:
            print(
                f"[ModelLoader] Note: AwqConfig(backend='{awq_backend}') unavailable ({exc}); "
                f"using checkpoint default config for '{model_name}'."
            )
            return None

    if target_quant != "bnb4" and not load_in_8bit:
        return None

    if importlib.util.find_spec("bitsandbytes") is None:
        print(
            "[ModelLoader] Note: 'bitsandbytes' not found in environment; "
            "falling back to standard dtype without 4-bit/8-bit weight quantization."
        )
        return None
    from transformers import BitsAndBytesConfig

    if target_quant == "bnb4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _load_tokenizer_cached(model_name: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    """Loads tokenizer from local disk cache first if already downloaded, else downloads."""
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )


def _load_causal_lm_cached(model_name: str, **kwargs) -> PreTrainedModel:
    """
    Checks if the model weights are already cached on local disk (`local_files_only=True`)
    to skip network requests; falls back to downloading from Hugging Face Hub only if missing.
    If `attn_implementation='flash_attention_2'` fails to load, raises `FlashAttentionUnavailableError`
    (no SDPA fallback).
    """
    def _try_load(load_kwargs: Dict[str, Any]) -> PreTrainedModel:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                **load_kwargs,
            )
            print(f"[ModelLoader] Loaded '{model_name}' directly from local disk cache.")
            return model
        except Exception:
            print(f"[ModelLoader] Downloading/loading '{model_name}' from Hugging Face Hub...")
            return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    attn_impl = kwargs.get("attn_implementation")
    try:
        model = _try_load(kwargs)
    except Exception as exc:
        if attn_impl == "flash_attention_2":
            raise FlashAttentionUnavailableError(
                f"Loading '{model_name}' with attn_implementation='flash_attention_2' failed "
                f"({type(exc).__name__}: {exc}). " + FLASH_ATTN_INSTALL_HINT
            ) from exc
        raise

    if attn_impl == "flash_attention_2":
        actual = getattr(model.config, "_attn_implementation", None)
        if actual != "flash_attention_2":
            raise FlashAttentionUnavailableError(
                f"'{model_name}' loaded with attn_implementation='{actual}' instead of "
                "'flash_attention_2'. " + FLASH_ATTN_INSTALL_HINT
            )
    return model


def load_model_pair(
    config: Optional[ModelConfig] = None,
    load_draft: bool = True,
    force_reload: bool = False,
) -> LoadedModels:
    """
    Loads the shared Qwen2.5 tokenizer, the draft model, and the target model onto the configured device.

    Supports Step 2 (`flash_attention_2` required on CUDA for both models; raises if unavailable)
    and Step 3 (`target_quant` in {fp16, bnb4, awq} without mixing Base and Instruct models).
    Automatically reuses already-loaded models in memory when called with identical configuration.
    """
    cfg = config or ModelConfig()
    device, dtype = resolve_device_and_dtype(cfg)
    if (
        device.type == "cuda"
        and (cfg.attn_implementation or "auto").lower() in _FA2_REQUESTS
        and dtype != torch.float16
    ):
        print(f"[ModelLoader] FlashAttention-2 enabled: using fp16 for both models (requested {dtype}).")
        dtype = torch.float16
    resolved_quant, target_name, draft_name = resolve_model_pair_for_quant(cfg)
    resolved_attn = resolve_attn_implementation(cfg.attn_implementation, dtype, device.type)

    cache_key = (
        draft_name if load_draft else None,
        target_name,
        str(device),
        str(dtype),
        resolved_quant,
        bool(cfg.load_target_in_8bit and device.type == "cuda"),
        bool(cfg.load_draft_in_8bit and device.type == "cuda"),
        resolved_attn,
        str(cfg.awq_backend),
        str(getattr(cfg, "draft_attn_implementation", "sdpa")),
    )

    if not force_reload and cache_key in _MODEL_CACHE:
        print(
            f"[ModelLoader] Reusing already-loaded models from memory "
            f"(Draft='{draft_name}', Target='{target_name}', quant={resolved_quant}, attn={resolved_attn})."
        )
        return _MODEL_CACHE[cache_key]

    # Also check if we previously loaded both Draft + Target and now only need Target
    if not force_reload and not load_draft:
        full_key = (draft_name,) + cache_key[1:]
        if full_key in _MODEL_CACHE:
            print(f"[ModelLoader] Reusing already-loaded Target model '{target_name}' from memory.")
            return _MODEL_CACHE[full_key]

    print(f"[ModelLoader] Loading tokenizer for '{target_name}'...")
    try:
        tokenizer = _load_tokenizer_cached(target_name, cfg.trust_remote_code)
    except Exception:
        tokenizer = _load_tokenizer_cached(draft_name, cfg.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if resolved_quant == "bnb4" and device.type == "cuda":
        print(
            "[ModelLoader] Note: bitsandbytes NF4 4-bit active on target model. "
            "Single-token decode uses fast NF4 kernels, but multi-token speculative verification "
            "(q_len > 1) falls back to dequant-to-FP16 (~1.5x TPOT)."
        )

    eff_quant = resolved_quant if device.type == "cuda" else "fp32"
    target_quant_cfg = _build_quantization_config(
        eff_quant,
        cfg.load_target_in_8bit and device.type == "cuda",
        dtype,
        awq_backend=cfg.awq_backend,
        model_name=target_name,
    )

    print(
        f"[ModelLoader] Initializing Target Model '{target_name}' "
        f"(device={device}, dtype={dtype}, attn={resolved_attn}, quant={eff_quant}, 8bit={cfg.load_target_in_8bit})..."
    )
    target_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "dtype": dtype,
        "attn_implementation": resolved_attn,
    }
    if target_quant_cfg is not None:
        target_kwargs["quantization_config"] = target_quant_cfg
        target_kwargs["device_map"] = "auto"
    elif device.type == "cuda":
        target_kwargs["device_map"] = "auto"

    try:
        target_model = _load_causal_lm_cached(target_name, **target_kwargs)
    except FlashAttentionUnavailableError:
        raise
    except Exception as exc:
        if eff_quant == "awq" and "quantization_config" in target_kwargs:
            print(
                f"[ModelLoader] Warning: AWQ load with backend='{cfg.awq_backend}' failed ({exc}); "
                "retrying with checkpoint default AWQ config..."
            )
            fallback_awq_kwargs = dict(target_kwargs)
            fallback_awq_kwargs.pop("quantization_config", None)
            target_model = _load_causal_lm_cached(target_name, **fallback_awq_kwargs)
        else:
            raise

    if "device_map" not in target_kwargs:
        target_model = target_model.to(device)
    target_model.eval()

    draft_model: Optional[PreTrainedModel] = None
    draft_attn = "none"
    if load_draft:
        draft_req = (getattr(cfg, "draft_attn_implementation", "sdpa") or "sdpa").lower()
        draft_load_attn = (
            resolve_attn_implementation("flash_attention_2", dtype, device.type)
            if draft_req in ("flash_attention_2", "fa2")
            else draft_req
        )
        draft_quant = _build_quantization_config(
            "fp16",
            cfg.load_draft_in_8bit and device.type == "cuda",
            dtype,
            model_name=draft_name,
        )
        print(
            f"[ModelLoader] Initializing Draft Model '{draft_name}' "
            f"(attn={draft_load_attn}; target stays on {resolved_attn})..."
        )
        draft_kwargs: Dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "dtype": dtype,
            "attn_implementation": draft_load_attn,
        }
        if draft_quant is not None:
            draft_kwargs["quantization_config"] = draft_quant
            draft_kwargs["device_map"] = "auto"
        elif device.type == "cuda":
            draft_kwargs["device_map"] = "auto"

        draft_model = _load_causal_lm_cached(draft_name, **draft_kwargs)
        if "device_map" not in draft_kwargs:
            draft_model = draft_model.to(device)
        draft_model.eval()
        draft_attn = use_packed_gqa_draft_attention(draft_model)

    actual_attn = str(getattr(target_model.config, "_attn_implementation", resolved_attn))
    check_vram_fit_for_sequence(target_model, draft_model, eff_quant, seq_len=16384)

    loaded = LoadedModels(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        attn_implementation=actual_attn,
        target_quant=eff_quant,
        draft_attn_implementation=draft_attn,
    )
    _MODEL_CACHE[cache_key] = loaded
    return loaded


class _SimpleTestTokenizer:
    """Deterministic lightweight tokenizer for offline unit testing and mock runs."""

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        words = text.strip().split()
        base_ids = [
            (abs(hash(w)) % (self.vocab_size - 10)) + 10 for w in words
        ]
        ids = [self.bos_token_id] + base_ids
        tensor = torch.tensor([ids], dtype=torch.long)

        class _Batch(dict):
            def __init__(self, input_ids: torch.Tensor):
                super().__init__(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
                self.input_ids = input_ids
                self.attention_mask = torch.ones_like(input_ids)

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                self["input_ids"] = self.input_ids
                self["attention_mask"] = self.attention_mask
                return self

        return _Batch(tensor)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return " ".join(f"tok_{tid}" for tid in token_ids if not (skip_special_tokens and tid in (0, 1, 2)))


def create_tiny_qwen_pair_for_testing(
    device: Optional[torch.device] = None,
    vocab_size: int = 512,
    force_reload: bool = False,
) -> LoadedModels:
    """
    Creates (or reuses from in-memory cache) a miniature pair of Qwen2ForCausalLM models
    (simulating Qwen2.5-0.5B draft and Qwen2.5-7B target architectures with GQA)
    for fast unit testing & dry-run verification.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_key = (str(dev), vocab_size)
    if not force_reload and mock_key in _MOCK_MODEL_CACHE:
        return _MOCK_MODEL_CACHE[mock_key]

    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    torch.manual_seed(42)
    draft_cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=1,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
        _attn_implementation="sdpa",
    )
    target_cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
        _attn_implementation="sdpa",
    )

    target_model = Qwen2ForCausalLM(target_cfg).to(device=dev, dtype=dtype).eval()
    draft_model = Qwen2ForCausalLM(draft_cfg).to(device=dev, dtype=dtype).eval()

    # Align draft model with target model's embeddings, layer 0, norm, and lm_head,
    # and scale down residual projections in target layers 1..7 so draft & target
    # share high predictive agreement (~80-90% acceptance rate, realistic for 0.5B -> 7B).
    with torch.no_grad():
        draft_model.model.embed_tokens.weight.copy_(target_model.model.embed_tokens.weight)
        draft_model.model.layers[0].load_state_dict(target_model.model.layers[0].state_dict())
        draft_model.model.norm.weight.copy_(target_model.model.norm.weight)
        draft_model.lm_head.weight.copy_(target_model.lm_head.weight)

        for layer in target_model.model.layers[1:]:
            layer.self_attn.o_proj.weight.mul_(0.04)
            layer.mlp.down_proj.weight.mul_(0.04)

    tokenizer = _SimpleTestTokenizer(vocab_size=vocab_size)

    loaded = LoadedModels(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device=dev,
        dtype=dtype,
        draft_attn_implementation=use_packed_gqa_draft_attention(draft_model),
    )
    _MOCK_MODEL_CACHE[mock_key] = loaded
    return loaded

"""
Configuration settings for Adaptive KV-Cache Compression & Speculative Decoding.

Default models:
  - Draft Model:  Qwen/Qwen2.5-0.5B
  - Target Model: Qwen/Qwen2.5-7B
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AdaptiveCacheConfig:
    """Configuration for the Adaptive KV-Cache Compression scheme."""

    # Enable/disable adaptive compression
    enabled: bool = True

    # Minimum/base KV token budget retained per layer (FP Hot Tier + Quantized Warm Tier)
    max_cache_size: int = 128

    # Optional proportional retention ratio (e.g., 0.25 = retain 25% of cumulative context).
    # When set, effective cache budget scales as max(max_cache_size, int(seen * cache_budget_ratio))
    # up to `max_cache_ceiling` (if set), preventing severe degradation on 4K-32K prompts.
    cache_budget_ratio: Optional[float] = None
    max_cache_ceiling: Optional[int] = None

    # StreamingLLM-style attention sink tokens (always kept in full precision at the start)
    num_sink_tokens: int = 4

    # Local recency window tokens (always kept in full precision at the end)
    recent_window_size: int = 32

    # Fraction of remaining mid-cache budget kept in full precision (Heavy Hitters).
    # Default 1.0 = warm-tier INT4/INT8 quantization OFF (pure pruning: exact budget in FP16, 4.00x at
    # 0.25 budget, no per-step dequant/cat). Set < 1.0 only to ablate mixed INT8-K/INT4-V warm tokens.
    heavy_hitter_ratio: float = 1.0

    # Observation window size (W_obs) used to compute true query-aware Q_obs @ K^T scores
    obs_window_size: int = 16

    # Quantization bit-width for low/medium-importance retained tokens:
    # `quant_bits` applies to Value states (per-token); `key_quant_bits` applies to Key states
    # (per-channel along seq_len to neutralize channel-wise bias outliers).
    quant_bits: int = 4
    key_quant_bits: int = 8

    # Exponential moving average decay for token importance scores across steps
    importance_decay: float = 0.95

    # Score bonus weight for recency when computing composite token relevance
    recency_weight: float = 0.1

    # Chunk size for long-prompt prefill (prevents OOM and enables streaming compression on 4K-32K prompts)
    prefill_chunk_size: int = 512

    # Step 6 (Memory / OOM): Compute LM-head logits only for positions needed (e.g., logits_to_keep=1 on prefill/decode)
    logits_to_keep: bool = True

    # Compress only when it can pay off: if the total cache length (prompt + max_new_tokens; for PPL: prefix +
    # scored continuation) is below this many tokens, the cache is NOT compressed at all (AdaptKV == Baseline,
    # Proposed == Spec-only, same code path). Decided once per run, so every doc at one sweep length agrees.
    compress_min_len: int = 4096


def cache_config_for_context(cfg: "AdaptiveCacheConfig", context_len: int) -> "AdaptiveCacheConfig":
    """Returns `cfg` with compression disabled when `context_len < cfg.compress_min_len` (else `cfg` unchanged)."""
    if cfg is not None and cfg.enabled and int(context_len) < int(getattr(cfg, "compress_min_len", 0) or 0):
        from dataclasses import replace

        return replace(cfg, enabled=False)
    return cfg


@dataclass
class SpeculativeConfig:
    """Configuration for the Speculative Decoding pipeline."""

    # Enable/disable speculative decoding
    enabled: bool = True

    # Number of lookahead tokens (gamma) proposed by the draft model per step
    gamma: int = 5

    # Enable dynamic/adaptive gamma adjustment based on draft model confidence
    # Disabled by default to avoid per-draft-token .item() CPU synchronization stalls
    adaptive_gamma: bool = False

    # Use pre-allocated StaticCache on draft model when supported for O(1) rollback and torch.compile
    use_static_draft_cache: bool = True

    # Compile draft model with torch.compile(..., mode="reduce-overhead") on CUDA to remove Python launch overhead
    # Compile / CUDA-graph runner options (Step 4)
    compile_draft: bool = False
    compile_target: bool = False
    target_use_static_cache: bool = False
    compile_mode: str = "reduce-overhead"

    # Draft cache consistency & keep-set sharing options (Step 5)
    consistent_draft_cache: bool = True
    draft_error_on_overflow: bool = True
    round_static_cache_to_8: bool = True
    # Draft KV cache when the target cache is compressed (Proposed):
    #   "shared_keepset": the draft keeps the same token positions as the target (target's consensus keep-set,
    #                     compacted in place inside the draft's StaticCache, CUDA graphs kept) -> the draft
    #                     attends to the same ~25% of the context as the target.
    #   "full":           the draft keeps the full uncompressed context.
    # Ignored when the target cache is not compressed (Spec-only, or context < compress_min_len).
    draft_cache: str = "shared_keepset"
    # Legacy override (stage ablation): True -> "shared_keepset", False -> "full", None -> use `draft_cache`.
    share_draft_keep_set: Optional[bool] = None

    # Timing synchronization mode: call torch.cuda.synchronize() before every draft/verify timer read.
    # Default OFF: throughput/TPOT are measured without the extra syncs. Benchmarks always run a separate
    # short pass with timing_mode=ON only to report the draft/verify split.
    timing_mode: bool = False

    # Minimum draft token probability required to continue drafting in the current step
    confidence_threshold: float = 0.35

    # Sampling temperature (0.0 = greedy verification, > 0.0 = Leviathan et al. rejection sampling)
    temperature: float = 0.0

    # Top-p (nucleus) sampling threshold when temperature > 0
    top_p: float = 0.9

    # Diagnostics only: profile one draft decode step (q_len=1) inside the generation (used by the d/v pass).
    profile_draft_step: bool = False

    def resolved_draft_cache(self) -> str:
        if self.share_draft_keep_set is not None:
            return "shared_keepset" if self.share_draft_keep_set else "full"
        return str(self.draft_cache)


@dataclass
class ModelConfig:
    """Configuration for Draft and Target Qwen2.5 models."""

    draft_model_name: str = "Qwen/Qwen2.5-0.5B"
    target_model_name: str = "Qwen/Qwen2.5-7B"

    # Device placement ("cuda", "cpu", or "auto")
    device: str = "auto"

    # Precision ("float16", "bfloat16", "float32")
    dtype: str = "float16"

    # Attention kernel implementation ("auto", "flash_attention_2", "sdpa", or "eager")
    # On CUDA, "auto" REQUIRES "flash_attention_2" (raises if `flash_attn` is missing; no SDPA fallback).
    # On CPU (mock/tests) "auto" resolves to "sdpa" since FA2 kernels need a GPU.
    attn_implementation: str = "auto"

    # Draft model attention ("sdpa" default, switched to packed-GQA decode after load). FA2 is not used for the
    # draft: on its StaticCache FA2 needs a padding mask -> per-layer host syncs + no CUDA-graph capture.
    draft_attn_implementation: str = "sdpa"

    # Target quantization mode ("fp16", "bnb4", or "awq"); default is FP16 base for L4 24GB
    target_quant: str = "fp16"
    awq_backend: str = "marlin"

    # Low-resource hardware weight quantization options (via bitsandbytes)
    load_target_in_4bit: bool = False
    load_target_in_8bit: bool = False
    load_draft_in_8bit: bool = False

    # Trust remote code if needed by custom tokenizers/models
    trust_remote_code: bool = True


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking across throughput, latency, perplexity, and energy."""

    max_new_tokens: int = 256
    num_warmup_runs: int = 1
    num_repeats: int = 1
    num_eval_docs: int = 5
    power_sample_interval_ms: int = 50
    fallback_gpu_tdp_watts: float = 115.0  # Consumer GPU fallback TDP (e.g., RTX 4060 Laptop/Desktop)
    fallback_cpu_tdp_watts: float = 45.0
    output_dir: str = "results"
    sweep_lengths: List[int] = field(default_factory=lambda: [1024, 8192, 16384])
    sweep_max_new_tokens: int = 256
    ppl_dataset: Optional[str] = None
    eval_continuation_tokens: int = 256
    batch_sweep_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    batch_sweep_lengths: List[int] = field(default_factory=lambda: [8192, 16384, 32768])
    stages: List[str] = field(default_factory=lambda: ["S0", "S1", "S2", "S3", "S4"])

    # Long-context prompts (~220+ tokens each) to evaluate KV-cache growth, memory bandwidth,
    # and speculative decoding acceleration under realistic context lengths.
    prompts: List[str] = field(
        default_factory=lambda: [
            (
                "Context: Large Language Models (LLMs) have demonstrated remarkable capabilities "
                "across natural language understanding, code generation, scientific discovery, and "
                "multi-step mathematical reasoning tasks. However, their real-world deployment in "
                "academic institutions, healthcare clinics, and public-service infrastructure is "
                "severely constrained by the high computational, memory bandwidth, and energy costs "
                "of autoregressive Transformer inference. During standard decoding, each generated "
                "token requires a sequential forward pass through billions of model parameters, and "
                "the growing Key-Value (KV) cache required for self-attention over long contexts "
                "consumes GPU memory that scales linearly with sequence length. Every decoding step "
                "must repeatedly read all cached Key and Value tensors from high-bandwidth memory, "
                "making generation strictly memory-bandwidth bound on consumer GPUs. By combining "
                "an adaptive KV-cache compression scheme—which preserves initial attention sink "
                "tokens, retains high-importance heavy-hitter tokens in full precision, quantizes "
                "medium-importance historical tokens to 4-bit integers, and prunes low-relevance "
                "tokens—with a speculative decoding pipeline where a fast draft model (Qwen2.5-0.5B) "
                "proposes multiple candidate tokens verified in parallel by the target model "
                "(Qwen2.5-7B), we can simultaneously reduce latency, VRAM usage, and energy per token.\n"
                "Question: Provide a detailed architectural analysis of how KV-cache pruning and "
                "speculative verification interact to minimize memory bandwidth bottlenecks:\n"
            ),
            (
                "Context: Sustainable computing and Green AI have become critical priorities aligned "
                "with United Nations Sustainable Development Goals SDG 9 (Industry, Innovation and "
                "Infrastructure) and SDG 12 (Responsible Consumption and Production). While training "
                "a foundation model is a one-time capital expenditure, inference dominates over 80 "
                "percent of the lifetime compute, electricity consumption, and carbon footprint of "
                "deployed generative AI systems. Organizations without access to expensive data-center "
                "accelerators must rely on commodity edge or consumer GPUs with limited VRAM capacity "
                "and strict thermal power envelopes. When serving long-context workloads such as "
                "document summarization, clinical record analysis, or multi-turn tutoring, the "
                "uncompressed FP16 KV cache rapidly saturates memory capacity and forces costly page "
                "swapping or small batch sizes. Algorithmic co-design that pairs dynamic token "
                "eviction and sub-byte quantization with draft-model lookahead speculation reduces "
                "both the number of full-model parameter reads and the volume of KV state transfers "
                "per emitted token.\n"
                "Question: Explain how reducing target forward passes and KV memory traffic lowers "
                "Joules per token and carbon emissions on consumer hardware:\n"
            ),
        ]
    )

    perplexity_texts: List[str] = field(
        default_factory=lambda: [
            (
                "Large Language Models have demonstrated remarkable capabilities across natural "
                "language understanding, code generation, and complex multi-step reasoning tasks. "
                "However, their real-world deployment is severely constrained by the high computational "
                "and memory cost of autoregressive inference. Each generated token requires a full "
                "forward pass through billions of parameters, and the growing Key-Value cache needed "
                "for attention over long contexts consumes memory that scales linearly with sequence "
                "length. By combining dynamic token pruning, mixed-precision cache quantization, and "
                "speculative execution with a lightweight draft model, inference systems can achieve "
                "significant reductions in latency and energy consumption per token without degrading "
                "predictive quality."
            )
        ]
    )

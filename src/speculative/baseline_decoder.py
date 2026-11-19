"""
Baseline Autoregressive Decoder for Qwen2.5-7B (with optional Adaptive KV-Cache Compression).
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StaticCache

from config import AdaptiveCacheConfig, cache_config_for_context
from src.kv_cache.adaptive_cache import AdaptiveKVCache
from src.speculative.static_runner import (
    ContextLengthOverflowError,
    get_peak_vram_mb,
    get_reusable_static_cache,
    model_forward_with_logits_to_keep,
    reset_peak_vram,
    round_up_to_multiple,
    run_bucketed_static_step,
    static_cache_memory_stats,
    sync_perf_counter,
)


@dataclass
class GenerationOutput:
    """Standardized output and telemetry from a generation run."""

    text: str
    output_ids: torch.Tensor
    num_generated_tokens: int
    total_time_s: float
    ttft_ms: float
    tpot_ms: float
    tokens_per_sec: float
    target_forward_passes: int
    draft_forward_passes: int = 0
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    mean_accepted_per_step: float = 0.0
    mean_drafted_per_step: float = 0.0
    decode_tokens_per_sec: float = 0.0
    draft_time_ms: float = 0.0
    draft_ms_per_token: float = 0.0
    draft_path: str = ""
    verify_time_ms: float = 0.0
    rollback_time_ms: float = 0.0
    cache_overhead_ms: float = 0.0
    peak_vram_mb: float = 0.0
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    draft_cache_stats: Dict[str, Any] = field(default_factory=dict)
    draft_diag: Dict[str, Any] = field(default_factory=dict)
    draft_cache_mode: str = ""


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Samples or argmaxes the next token from 2D logits of shape (batch, vocab_size)."""
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, dim=-1, index=next_in_sorted)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def autoregressive_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: Union[str, List[str]],
    max_new_tokens: int = 128,
    cache_config: Optional[AdaptiveCacheConfig] = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    stop_on_eos: bool = True,
    use_static_cache: bool = False,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    round_static_cache_to_8: bool = True,
    logits_to_keep: Optional[bool] = None,
    timing_mode: bool = False,
    error_on_overflow: bool = True,
) -> GenerationOutput:
    """
    Runs standard token-by-token autoregressive decoding on the target LLM,
    optionally applying Adaptive KV-Cache Pruning & Mixed-Precision Quantization
    or StaticCache + torch.compile CUDA graphs (Step 4).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("Prompt list must not be empty.")
        tok_seqs = [tokenizer(p, return_tensors="pt")["input_ids"] for p in prompt]
        min_len = min(t.shape[1] for t in tok_seqs)
        input_ids = torch.cat([t[:, -min_len:] for t in tok_seqs], dim=0).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

    batch_size = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])

    # --compress-min-len: short contexts run uncompressed (AdaptKV == Baseline code path)
    cfg = cache_config_for_context(cache_config or AdaptiveCacheConfig(enabled=False), seq_len + max(0, int(max_new_tokens)))
    use_ltk = cfg.logits_to_keep if logits_to_keep is None else bool(logits_to_keep)
    ltk_count = 1 if use_ltk else None

    use_static = bool(use_static_cache and not cfg.enabled)
    max_needed_static = seq_len + max_new_tokens + 8
    max_pos_emb = int(getattr(model.config, "max_position_embeddings", max_needed_static))
    if use_static and max_needed_static > max_pos_emb:
        if error_on_overflow:
            raise ContextLengthOverflowError(
                f"StaticCache length {max_needed_static} exceeds model max_position_embeddings={max_pos_emb}."
            )
        use_static = False

    if use_static:
        cache_cap = round_up_to_multiple(max_needed_static, 8) if round_static_cache_to_8 else max_needed_static
        kv_cache: Any = (
            get_reusable_static_cache(model, cache_cap, batch_size=batch_size)
            if compile_model
            else StaticCache(config=model.config, max_cache_len=cache_cap)
        )
    else:
        kv_cache = AdaptiveKVCache(config=cfg)

    generated_tensors: List[torch.Tensor] = []
    target_passes = 0

    if max_new_tokens <= 0:
        return GenerationOutput(
            text="",
            output_ids=input_ids,
            num_generated_tokens=0,
            total_time_s=0.0,
            ttft_ms=0.0,
            tpot_ms=0.0,
            tokens_per_sec=0.0,
            target_forward_passes=0,
            peak_vram_mb=get_peak_vram_mb(device),
            cache_stats=(
                static_cache_memory_stats(kv_cache, seq_len, device, dtype, "StaticCache")
                if use_static
                else kv_cache.get_memory_stats()
            ),
        )

    if not use_static and cfg.enabled:
        kv_cache.register_query_hooks(model)

    reset_peak_vram(device)
    try:
        t_start = sync_perf_counter(device, timing_mode or device.type == "cuda")

        # 1. Prefill Phase (chunked to prevent peak VRAM OOM on 4K-32K contexts)
        chunk_size = max(1, cfg.prefill_chunk_size) if seq_len > cfg.prefill_chunk_size else seq_len
        last_logits: Optional[torch.Tensor] = None
        seen_so_far = 0
        for start in range(0, seq_len, chunk_size):
            chunk = input_ids[:, start : start + chunk_size]
            if use_static:
                c_logits = run_bucketed_static_step(
                    model=model,
                    past_key_values=kv_cache,
                    input_ids=chunk,
                    cur_seen=seen_so_far,
                    logits_to_keep_count=1 if use_ltk else 0,
                    compile_enabled=False,
                )
                last_logits = c_logits[:, -1, :]
            else:
                out = model_forward_with_logits_to_keep(
                    model,
                    logits_to_keep_count=ltk_count,
                    input_ids=chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                last_logits = out.logits[:, -1, :]
                if cfg.enabled and kv_cache.should_compress():
                    kv_cache.compress_step()
            seen_so_far += int(chunk.shape[1])
            target_passes += 1
        if not use_static:
            kv_cache.compress_step(force=True)

        next_token = _sample_from_logits(last_logits, temperature=temperature, top_p=top_p)
        ttft_ms = (sync_perf_counter(device, timing_mode or device.type == "cuda") - t_start) * 1000.0
        generated_tensors.append(next_token)

        check_eos = stop_on_eos and batch_size == 1 and tokenizer.eos_token_id is not None

        # 2. Autoregressive Decode Loop
        cache_overhead_s = 0.0
        decode_steps = 0
        for _ in range(max_new_tokens - 1):
            if check_eos and int(next_token.item()) == tokenizer.eos_token_id:
                break

            if use_static:
                step_logits = run_bucketed_static_step(
                    model=model,
                    past_key_values=kv_cache,
                    input_ids=next_token,
                    cur_seen=seen_so_far,
                    logits_to_keep_count=1 if use_ltk else 0,
                    compile_enabled=compile_model,
                    compile_mode=compile_mode,
                )
                last_step_logits = step_logits[:, -1, :]
            else:
                out = model_forward_with_logits_to_keep(
                    model,
                    logits_to_keep_count=ltk_count,
                    input_ids=next_token,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                last_step_logits = out.logits[:, -1, :]
                t_c0 = sync_perf_counter(device, timing_mode)
                kv_cache.compress_step()
                cache_overhead_s += sync_perf_counter(device, timing_mode) - t_c0

            seen_so_far += 1
            target_passes += 1
            decode_steps += 1

            next_token = _sample_from_logits(
                last_step_logits, temperature=temperature, top_p=top_p
            )
            generated_tensors.append(next_token)

        if not use_static:
            t_c1 = sync_perf_counter(device, timing_mode)
            kv_cache.compress_step(force=True)
            cache_overhead_s += sync_perf_counter(device, timing_mode) - t_c1
        total_time_s = max(1e-6, sync_perf_counter(device, timing_mode or device.type == "cuda") - t_start)
    finally:
        if not use_static:
            kv_cache.remove_query_hooks()

    peak_vram_mb = get_peak_vram_mb(device)
    gen_tensor = torch.cat(generated_tensors, dim=-1)  # (batch_size, gen_steps)
    gen_steps = int(gen_tensor.shape[1])
    num_gen = gen_steps * batch_size
    decode_time_ms = max(0.0, total_time_s * 1000.0 - ttft_ms)
    tpot_ms = decode_time_ms / max(1, gen_steps - 1) if gen_steps > 1 else ttft_ms
    tokens_per_sec = num_gen / total_time_s
    decode_tokens_per_sec = (
        ((gen_steps - 1) * batch_size * 1000.0 / decode_time_ms)
        if gen_steps > 1 and decode_time_ms > 0
        else tokens_per_sec
    )
    cache_overhead_ms = (cache_overhead_s * 1000.0) / max(1, decode_steps)

    full_ids = torch.cat([input_ids, gen_tensor], dim=-1)
    decoded_text = tokenizer.decode(gen_tensor[0].tolist(), skip_special_tokens=True)

    c_stats = (
        static_cache_memory_stats(kv_cache, seen_so_far, device, dtype, "StaticCache")
        if use_static
        else kv_cache.get_memory_stats()
    )

    return GenerationOutput(
        text=decoded_text,
        output_ids=full_ids,
        num_generated_tokens=num_gen,
        total_time_s=round(total_time_s, 4),
        ttft_ms=round(ttft_ms, 2),
        tpot_ms=round(tpot_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        target_forward_passes=target_passes,
        draft_forward_passes=0,
        total_drafted_tokens=0,
        total_accepted_tokens=0,
        acceptance_rate=0.0,
        mean_accepted_per_step=1.0,
        decode_tokens_per_sec=round(decode_tokens_per_sec, 2),
        draft_time_ms=0.0,
        verify_time_ms=round(tpot_ms, 2),
        cache_overhead_ms=round(cache_overhead_ms, 3),
        peak_vram_mb=peak_vram_mb,
        cache_stats=c_stats,
        draft_cache_stats={},
    )

"""
Speculative Decoding Pipeline with Adaptive KV-Cache Compression.

Pairs a lightweight draft model (Qwen/Qwen2.5-0.5B) with a high-capacity target model
(Qwen/Qwen2.5-7B). The draft model proposes `gamma` lookahead tokens autoregressively,
and the target model verifies all `gamma` tokens in a single parallel forward pass.

Features:
- Lossless greedy verification & stochastic rejection sampling (Leviathan et al., 2023)
- Vocabulary alignment handling (e.g. Qwen2.5-0.5B 151936 vs Qwen2.5-7B 152064)
- Synchronized KV-cache rollback (`crop`) across both draft and target models
- Dynamic confidence-aware lookahead (`adaptive_gamma`) to avoid wasted draft steps
- Integrated Adaptive KV-Cache Pruning & Quantization after verification
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StaticCache

from config import AdaptiveCacheConfig, SpeculativeConfig, cache_config_for_context
from src.kv_cache.adaptive_cache import AdaptiveKVCache
from src.speculative.baseline_decoder import GenerationOutput
from src.speculative.static_runner import (
    ContextLengthOverflowError,
    compact_static_cache,
    crop_static_cache,
    static_cache_is_static_address,
    get_peak_vram_mb,
    get_reusable_static_cache,
    model_forward_with_logits_to_keep,
    reset_peak_vram,
    round_up_to_multiple,
    run_bucketed_static_step,
    draft_path_label,
    static_cache_bucket,
    static_cache_memory_stats,
    sync_perf_counter,
)


# Module-level cache of compiled dynamic target wrappers for AdaptiveKVCache paths
_COMPILED_DYNAMIC_TARGET: Dict[int, Any] = {}


def _get_dynamic_target_runner(
    target_model: PreTrainedModel, compile_target: bool, device: torch.device
) -> Any:
    """Returns a `dynamic=True` compiled wrapper on CUDA when AdaptiveKV pruning changes KV shape."""
    if not (compile_target and device.type == "cuda"):
        return target_model
    key = id(target_model)
    if key in _COMPILED_DYNAMIC_TARGET:
        cached = _COMPILED_DYNAMIC_TARGET[key]
        return cached if cached is not None else target_model
    try:
        compiled = torch.compile(target_model, dynamic=True, fullgraph=False)
        _COMPILED_DYNAMIC_TARGET[key] = compiled
        return compiled
    except Exception:
        _COMPILED_DYNAMIC_TARGET[key] = None
        return target_model


def _profile_draft_decode_step(
    draft_model: PreTrainedModel,
    draft_cache: Any,
    ids: torch.Tensor,
    logical: int,
    evicted: int,
    step_fn: Any,
    compiled: bool,
    repeats: int = 5,
) -> Dict[str, Any]:
    """
    Diagnostics for ONE draft decode step (q_len=1) at the current context. Re-runs the step at the same slot
    (eager once, then the real compiled/CUDA-graph path `repeats` times), restoring the cache cursor after each,
    so generation is unaffected (the slot is overwritten by the real step right after). Reports: attention path,
    mask shape vs static capacity, whether the KV buffers are static addresses (no CUDA-graph input copy),
    per-step time, top GPU kernels, and copy traffic during the step.
    """
    from torch.profiler import ProfilerActivity, profile
    from src.speculative.draft_attention import LAST_DECODE_INFO

    dev = ids.device
    phys = logical - evicted
    layer0 = draft_cache.layers[0]
    info: Dict[str, Any] = {
        "logical_len": int(logical),
        "physical_len": int(phys),
        "static_capacity": int(getattr(layer0, "max_cache_len", 0)),
        "kv_static_address": static_cache_is_static_address(draft_cache),
        "attn_impl": str(getattr(draft_model.config, "_attn_implementation", "?")),
    }

    def _sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def _top(prof: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        on_cuda = dev.type == "cuda"
        evts = list(prof.key_averages())
        key = (lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))) if on_cuda \
            else (lambda e: e.self_cpu_time_total)
        top = [
            {"name": e.key[:90], "calls": int(e.count), "ms": round(key(e) / 1000.0, 3)}
            for e in sorted(evts, key=key, reverse=True)[:6]
            if key(e) > 0
        ]
        copies = [e for e in evts if ("memcpy" in e.key.lower() or e.key in ("aten::copy_", "Memcpy DtoD (Device -> Device)"))]
        cp = {"copy_ops": int(sum(e.count for e in copies)), "copy_ms": round(sum(key(e) for e in copies) / 1000.0, 3)}
        return top, cp

    # 1) Eager step: records the attention path + mask shape (python attention code does not run on replay).
    LAST_DECODE_INFO.clear()
    act = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with profile(activities=act) as prof_e:
        step_fn(ids, logical, False)
        _sync()
    crop_static_cache(draft_cache, phys)
    info["eager_attention"] = dict(LAST_DECODE_INFO)
    info["eager_top_kernels"], _ = _top(prof_e)

    # 2) The real (compiled / CUDA-graph) path, timed and profiled.
    import time as _time

    step_fn(ids, logical, compiled)  # make sure graph is captured for this bucket
    _sync()
    crop_static_cache(draft_cache, phys)
    t0 = _time.perf_counter()
    for _ in range(repeats):
        step_fn(ids, logical, compiled)
        crop_static_cache(draft_cache, phys)
    _sync()
    info["step_ms"] = round((_time.perf_counter() - t0) * 1000.0 / repeats, 3)
    with profile(activities=act) as prof_c:
        step_fn(ids, logical, compiled)
        _sync()
    crop_static_cache(draft_cache, phys)
    info["compiled_top_kernels"], info["compiled_copies"] = _top(prof_c)
    info["path"] = draft_path_label(draft_model, True, compiled)
    return info


def _crop_cache(cache: Any, max_length: int, is_static: bool) -> None:
    if is_static:
        crop_static_cache(cache, max_length)
    else:
        cache.crop(max_length)


def _compute_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    shared_vocab_size: int,
) -> torch.Tensor:
    """Computes normalized probability distributions restricted to the shared vocabulary."""
    sliced = logits[..., :shared_vocab_size].float()
    if temperature <= 1e-5:
        probs = torch.zeros_like(sliced)
        argmax_idx = torch.argmax(sliced, dim=-1, keepdim=True)
        probs.scatter_(-1, argmax_idx, 1.0)
        return probs

    scaled = sliced / temperature
    probs = torch.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_indices, sorted_probs)
        return filtered
    return probs


@torch.no_grad()
def speculative_generate(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    spec_config: Optional[SpeculativeConfig] = None,
    target_cache_config: Optional[AdaptiveCacheConfig] = None,
    draft_cache_config: Optional[AdaptiveCacheConfig] = None,
    stop_on_eos: bool = True,
) -> GenerationOutput:
    """
    Executes speculative decoding with synchronized KV-cache rollback and
    adaptive KV-cache compression on both draft and target models.
    """
    s_cfg = spec_config or SpeculativeConfig()
    t_cache_cfg = target_cache_config or AdaptiveCacheConfig(enabled=True)
    greedy = s_cfg.temperature <= 1e-5
    use_ltk = bool(getattr(t_cache_cfg, "logits_to_keep", True))

    device = next(target_model.parameters()).device
    dtype = next(target_model.parameters()).dtype
    draft_device = next(draft_model.parameters()).device
    draft_dtype = next(draft_model.parameters()).dtype

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    # --compress-min-len: below it the target cache is not compressed at all, so Proposed runs the exact
    # Spec-only code path (full draft StaticCache, uncompressed target).
    t_cache_cfg = cache_config_for_context(t_cache_cfg, prompt_len + max(0, int(max_new_tokens)))
    # Keep draft cache in fast FP16 sparse pruning (`heavy_hitter_ratio=1.0`, zero INT4/INT8
    # dequantization inside the multi-token draft loop) while matching target's eviction budget.
    d_cache_cfg = draft_cache_config or replace(t_cache_cfg, heavy_hitter_ratio=1.0)

    max_needed_static = prompt_len + max_new_tokens + s_cfg.gamma + 16
    target_max_pos = int(getattr(target_model.config, "max_position_embeddings", max_needed_static))
    draft_max_pos = int(getattr(draft_model.config, "max_position_embeddings", max_needed_static))

    # Step 4: Optional StaticCache + compiled bucketed runner on target when AdaptiveKV is disabled
    use_static_target = bool((not t_cache_cfg.enabled) and s_cfg.target_use_static_cache)
    if use_static_target and max_needed_static > target_max_pos:
        if s_cfg.consistent_draft_cache and s_cfg.draft_error_on_overflow:
            raise ContextLengthOverflowError(
                f"Target StaticCache length {max_needed_static} exceeds max_position_embeddings={target_max_pos}."
            )
        use_static_target = False

    if use_static_target:
        t_cap = (
            round_up_to_multiple(max_needed_static, 8)
            if s_cfg.round_static_cache_to_8
            else max_needed_static
        )
        target_cache: Any = (
            get_reusable_static_cache(target_model, t_cap, batch_size=1)
            if s_cfg.compile_target
            else StaticCache(config=target_model.config, max_cache_len=t_cap)
        )
        target_dynamic_runner = target_model
    else:
        target_cache = AdaptiveKVCache(config=t_cache_cfg)
        if t_cache_cfg.enabled:
            target_cache.register_query_hooks(target_model)
        target_dynamic_runner = _get_dynamic_target_runner(target_model, s_cfg.compile_target, device)

    # Step 5: Draft cache consistency across context lengths + optional keep-set sharing
    use_shared_keepset = bool(
        s_cfg.resolved_draft_cache() == "shared_keepset" and t_cache_cfg.enabled and draft_cache_config is None
    )
    # Shared keep-set on the draft's StaticCache (compacted in place, CUDA graphs kept) unless the static draft
    # cache is disabled, in which case the legacy eager AdaptiveKVCache draft is used.
    use_shared_static_draft = bool(use_shared_keepset and s_cfg.use_static_draft_cache)
    use_shared_adaptive_draft = bool(use_shared_keepset and not s_cfg.use_static_draft_cache)
    wants_static_draft = bool(
        s_cfg.use_static_draft_cache
        and draft_cache_config is None
        and not use_shared_adaptive_draft
    )
    if wants_static_draft and max_needed_static > draft_max_pos:
        if s_cfg.consistent_draft_cache:
            if s_cfg.draft_error_on_overflow:
                raise ContextLengthOverflowError(
                    f"Draft StaticCache length {max_needed_static} exceeds draft model "
                    f"max_position_embeddings={draft_max_pos} (consistent_draft_cache=True; refusing silent switch)."
                )
            wants_static_draft = False
        else:
            wants_static_draft = False

    use_static_draft = wants_static_draft
    if use_static_draft:
        d_needed = max_needed_static
        if use_shared_static_draft:
            # Physical slots needed: the target's hot-tier budget at the final length + one amortization stride
            # + the largest block written between two compressions (prefill chunk or one spec step), + margin.
            _, _, _, fp_b, stride = target_cache._effective_budgets(max_needed_static)
            pf_chunk = max(1, min(max(1, prompt_len - 1), t_cache_cfg.prefill_chunk_size))
            d_needed = min(max_needed_static, fp_b + stride + max(pf_chunk, s_cfg.gamma + 2) + 64)
        if s_cfg.compile_draft:
            # One pooled StaticCache + CUDA graph per capacity bucket (multiple of 8), so per-document
            # prompt-length differences don't trigger a recompile / new graph capture.
            d_cap = static_cache_bucket(d_needed, draft_max_pos, s_cfg.round_static_cache_to_8)
        else:
            d_cap = (
                round_up_to_multiple(d_needed, 8)
                if s_cfg.round_static_cache_to_8
                else d_needed
            )
        draft_cache: Any = (
            get_reusable_static_cache(draft_model, d_cap, batch_size=1)
            if s_cfg.compile_draft
            else StaticCache(config=draft_model.config, max_cache_len=d_cap)
        )
    else:
        draft_cache = AdaptiveKVCache(config=d_cache_cfg)

    cur_target_pos = 0
    cur_draft_pos = 0  # logical draft position (RoPE)
    # Static draft only: tokens evicted from the draft buffer, so physical slot = cur_draft_pos - draft_evicted.
    draft_evicted = 0
    draft_diag: Dict[str, Any] = {}

    def _draft_static_step(ids: torch.Tensor, logical: int, compiled: bool) -> torch.Tensor:
        return run_bucketed_static_step(
            model=draft_model,
            past_key_values=draft_cache,
            input_ids=ids,
            cur_seen=logical - draft_evicted,
            logits_to_keep_count=1 if use_ltk else 0,
            compile_enabled=compiled,
            compile_mode=s_cfg.compile_mode,
            logical_pos=logical,
        )

    def _shared_compress(force: bool) -> int:
        """Compress the target; mirror its consensus keep-set into the draft. Returns draft tokens evicted."""
        if not use_shared_static_draft:
            target_cache.compress_step(force=force)
            draft_cache.compress_step(force=True, shared_mid_scores=target_cache.get_shared_mid_scores())
            return 0
        ev_before = target_cache.compress_events
        pre_len = int(target_cache.key_cache[0].shape[-2]) if len(target_cache.key_cache) else 0
        target_cache.compress_step(force=force)
        if target_cache.compress_events == ev_before:
            return 0
        draft_phys = int(draft_cache.get_seq_length())
        if draft_phys != pre_len:
            raise RuntimeError(
                f"shared_keepset lockstep broken: draft holds {draft_phys} tokens, target hot tier {pre_len}."
            )
        keep_idx = target_cache.get_shared_keep_indices()
        if keep_idx is None:
            raise RuntimeError("shared_keepset: target compressed but no shared keep-set is available.")
        kept = compact_static_cache(draft_cache, keep_idx)
        return pre_len - kept

    shared_vocab = min(
        int(draft_model.config.vocab_size),
        int(target_model.config.vocab_size),
    )

    generated_tokens: List[int] = []
    target_passes = 0
    draft_passes = 0
    total_drafted = 0
    total_accepted = 0
    spec_steps = 0
    draft_time_s = 0.0
    verify_time_s = 0.0
    rollback_time_s = 0.0
    cache_overhead_s = 0.0

    reset_peak_vram(device)
    try:
        t_start = sync_perf_counter(device, s_cfg.timing_mode or device.type == "cuda")
        ttft_ms: Optional[float] = None

        # 1. Prefill Phase
        if prompt_len > 1:
            prefix_ids = input_ids[:, :-1]
            prefix_len = int(prefix_ids.shape[1])
            chunk_size = (
                max(1, t_cache_cfg.prefill_chunk_size)
                if prefix_len > t_cache_cfg.prefill_chunk_size
                else prefix_len
            )
            d_prefix_ids = prefix_ids.to(draft_device)
            if use_shared_keepset:
                # Lockstep chunked prefill ensures Target and Draft share identical mid_len boundaries
                for start in range(0, prefix_len, chunk_size):
                    t_chunk = prefix_ids[:, start : start + chunk_size]
                    t_len = int(t_chunk.shape[1])
                    _ = model_forward_with_logits_to_keep(
                        target_model,
                        logits_to_keep_count=1 if use_ltk else None,
                        input_ids=t_chunk,
                        past_key_values=target_cache,
                        use_cache=True,
                    )
                    cur_target_pos += t_len
                    target_passes += 1

                    d_chunk = d_prefix_ids[:, start : start + chunk_size]
                    if use_shared_static_draft:
                        _ = _draft_static_step(d_chunk, cur_draft_pos, compiled=False)
                    else:
                        _ = model_forward_with_logits_to_keep(
                            draft_model,
                            logits_to_keep_count=1 if use_ltk else None,
                            input_ids=d_chunk,
                            past_key_values=draft_cache,
                            use_cache=True,
                        )
                    cur_draft_pos += t_len
                    draft_passes += 1

                    if t_cache_cfg.enabled and target_cache.should_compress():
                        draft_evicted += _shared_compress(force=False)
                draft_evicted += _shared_compress(force=True)
            else:
                for start in range(0, prefix_len, chunk_size):
                    t_chunk = prefix_ids[:, start : start + chunk_size]
                    t_len = int(t_chunk.shape[1])
                    if use_static_target:
                        _ = run_bucketed_static_step(
                            model=target_model,
                            past_key_values=target_cache,
                            input_ids=t_chunk,
                            cur_seen=cur_target_pos,
                            logits_to_keep_count=1 if use_ltk else 0,
                            compile_enabled=False,
                        )
                    else:
                        _ = model_forward_with_logits_to_keep(
                            target_model,
                            logits_to_keep_count=1 if use_ltk else None,
                            input_ids=t_chunk,
                            past_key_values=target_cache,
                            use_cache=True,
                        )
                        if t_cache_cfg.enabled and target_cache.should_compress():
                            target_cache.compress_step()
                    cur_target_pos += t_len
                    target_passes += 1
                if not use_static_target:
                    target_cache.compress_step(force=True)

                d_chunk_size = (
                    prefix_len
                    if (use_static_draft and prefix_len <= 4096)
                    else max(2048, d_cache_cfg.prefill_chunk_size)
                )
                for start in range(0, prefix_len, d_chunk_size):
                    d_chunk = d_prefix_ids[:, start : start + d_chunk_size]
                    d_len = int(d_chunk.shape[1])
                    if use_static_draft:
                        _ = run_bucketed_static_step(
                            model=draft_model,
                            past_key_values=draft_cache,
                            input_ids=d_chunk,
                            cur_seen=cur_draft_pos,
                            logits_to_keep_count=1 if use_ltk else 0,
                            compile_enabled=False,
                        )
                        cur_draft_pos += d_len
                    else:
                        _ = model_forward_with_logits_to_keep(
                            draft_model,
                            logits_to_keep_count=1 if use_ltk else None,
                            input_ids=d_chunk,
                            past_key_values=draft_cache,
                            use_cache=True,
                        )
                        cur_draft_pos += d_len
                        if d_cache_cfg.enabled and draft_cache.should_compress():
                            draft_cache.compress_step()
                    draft_passes += 1
                if not use_static_draft:
                    draft_cache.compress_step(force=True)

        last_token = input_ids[:, -1:]  # Shape: (1, 1)
        pending_draft_input = last_token.to(draft_device)

        # 2. Speculative Decoding Loop
        while len(generated_tokens) < max_new_tokens:
            spec_steps += 1
            remaining_budget = max_new_tokens - len(generated_tokens)
            current_gamma = min(s_cfg.gamma, remaining_budget)

            pre_step_target_len = int(target_cache.get_seq_length())
            # Note: if previous step accepted all gamma tokens, `pending_draft_input` has 2 tokens
            # ([d_gamma, bonus_token]), where `d_gamma` catches up the draft cache to match target_cache.
            catchup_offset = int(pending_draft_input.shape[1]) - 1
            # Logical length. Static draft: cur_draft_pos already equals the cache's logical length, and reading
            # cumulative_length (a device tensor) here would force a GPU sync every step.
            pre_step_draft_len = (
                cur_draft_pos + catchup_offset
                if use_static_draft
                else int(draft_cache.get_seq_length()) + catchup_offset
            )

            draft_tokens: List[int] = []
            draft_gpu_tokens: List[torch.Tensor] = []
            draft_probs_list: List[torch.Tensor] = []
            need_draft_step_sync = (not greedy) or s_cfg.adaptive_gamma

            # --- Step 2a: Draft Model Autoregressive Lookahead ---
            t_d0 = sync_perf_counter(draft_device, s_cfg.timing_mode)
            cur_draft_in = pending_draft_input
            initial_step_conf: Optional[float] = None
            for step_idx in range(current_gamma):
                q_len = int(cur_draft_in.shape[1])
                if use_static_draft:
                    if s_cfg.profile_draft_step and not draft_diag and q_len == 1 and spec_steps >= 2:
                        try:
                            draft_diag = _profile_draft_decode_step(
                                draft_model, draft_cache, cur_draft_in, cur_draft_pos, draft_evicted,
                                _draft_static_step, s_cfg.compile_draft,
                            )
                        except Exception as exc:  # diagnostics must never break a run
                            crop_static_cache(draft_cache, cur_draft_pos - draft_evicted)
                            draft_diag = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                    d_logits_full = _draft_static_step(cur_draft_in, cur_draft_pos, s_cfg.compile_draft)
                    d_logits = d_logits_full[:, -1, :shared_vocab]
                    cur_draft_pos += q_len
                else:
                    d_out = model_forward_with_logits_to_keep(
                        draft_model,
                        logits_to_keep_count=1 if use_ltk else None,
                        input_ids=cur_draft_in,
                        past_key_values=draft_cache,
                        use_cache=True,
                    )
                    d_logits = d_out.logits[:, -1, :shared_vocab]
                    cur_draft_pos += q_len
                draft_passes += 1

                if greedy:
                    d_tok_t = torch.argmax(d_logits, dim=-1, keepdim=True)  # (1, 1) on draft_device
                    draft_gpu_tokens.append(d_tok_t)
                    cur_draft_in = d_tok_t
                    if not need_draft_step_sync:
                        continue
                    d_tok = int(d_tok_t.item())
                    draft_tokens.append(d_tok)
                else:
                    q_probs = _compute_probs(
                        d_logits, s_cfg.temperature, s_cfg.top_p, shared_vocab
                    )  # (1, shared_vocab)
                    d_tok_t = torch.multinomial(q_probs, num_samples=1)
                    d_tok = int(d_tok_t.item())
                    draft_gpu_tokens.append(d_tok_t)
                    draft_tokens.append(d_tok)
                    draft_probs_list.append(q_probs.to(device))
                    cur_draft_in = d_tok_t

                # Dynamic/Adaptive Gamma: stop drafting early if draft model confidence drops
                if s_cfg.adaptive_gamma:
                    d_logits_f32 = d_logits.float()
                    raw_conf = float(
                        torch.exp(d_logits_f32[0, d_tok] - torch.logsumexp(d_logits_f32, dim=-1)[0]).item()
                    )
                    if initial_step_conf is None:
                        initial_step_conf = raw_conf
                    elif step_idx >= 1 and raw_conf < s_cfg.confidence_threshold * initial_step_conf:
                        break
            draft_time_s += sync_perf_counter(draft_device, s_cfg.timing_mode) - t_d0

            actual_gamma = len(draft_gpu_tokens)
            total_drafted += actual_gamma

            # --- Step 2b: Parallel Target Model Verification ---
            t_v0 = sync_perf_counter(device, s_cfg.timing_mode)
            if actual_gamma > 0:
                draft_tensor = torch.cat(
                    [t.to(device) if t.device != device else t for t in draft_gpu_tokens], dim=-1
                )
                verify_ids = torch.cat([last_token, draft_tensor], dim=-1)  # Shape: (1, actual_gamma + 1)
            else:
                draft_tensor = None
                verify_ids = last_token

            verify_q_len = int(verify_ids.shape[1])
            if use_static_target:
                verify_logits = run_bucketed_static_step(
                    model=target_model,
                    past_key_values=target_cache,
                    input_ids=verify_ids,
                    cur_seen=pre_step_target_len,
                    logits_to_keep_count=verify_q_len if use_ltk else 0,
                    compile_enabled=s_cfg.compile_target,
                    compile_mode=s_cfg.compile_mode,
                )
            else:
                try:
                    t_out = model_forward_with_logits_to_keep(
                        target_dynamic_runner,
                        logits_to_keep_count=verify_q_len if use_ltk else None,
                        input_ids=verify_ids,
                        past_key_values=target_cache,
                        use_cache=True,
                    )
                except Exception:
                    if target_dynamic_runner is not target_model:
                        _COMPILED_DYNAMIC_TARGET[id(target_model)] = None
                        target_dynamic_runner = target_model
                        t_out = model_forward_with_logits_to_keep(
                            target_model,
                            logits_to_keep_count=verify_q_len if use_ltk else None,
                            input_ids=verify_ids,
                            past_key_values=target_cache,
                            use_cache=True,
                        )
                    else:
                        raise
                verify_logits = t_out.logits
            target_passes += 1

            if ttft_ms is None:
                ttft_ms = (sync_perf_counter(device, s_cfg.timing_mode or device.type == "cuda") - t_start) * 1000.0

            # verify_logits has shape (1, actual_gamma + 1, vocab_size)
            accepted_in_step = 0
            final_emitted_token: Optional[int] = None

            # --- Step 2c: Rejection Sampling / Greedy Verification ---
            if greedy:
                target_all = (
                    torch.argmax(verify_logits[:, : actual_gamma + 1, :shared_vocab], dim=-1)[0].tolist()
                )
                if not draft_tokens and actual_gamma > 0 and draft_tensor is not None:
                    draft_tokens = draft_tensor[0].tolist()
                for i, cand_tok in enumerate(draft_tokens):
                    target_tok = int(target_all[i])
                    if cand_tok == target_tok:
                        accepted_in_step += 1
                        generated_tokens.append(cand_tok)
                        if (
                            stop_on_eos
                            and tokenizer.eos_token_id is not None
                            and cand_tok == tokenizer.eos_token_id
                        ):
                            break
                    else:
                        final_emitted_token = target_tok
                        break
                bonus_tok = int(target_all[actual_gamma])
            else:
                rand_u = (
                    torch.rand(actual_gamma, device=device).tolist() if actual_gamma > 0 else []
                )
                for i in range(actual_gamma):
                    p_probs = _compute_probs(
                        verify_logits[:, i, :], s_cfg.temperature, s_cfg.top_p, shared_vocab
                    )
                    q_probs = draft_probs_list[i]
                    cand_tok = draft_tokens[i]
                    p_val = p_probs[0, cand_tok].item()
                    q_val = max(1e-8, q_probs[0, cand_tok].item())
                    accept_prob = min(1.0, p_val / q_val)
                    if rand_u[i] <= accept_prob:
                        accepted_in_step += 1
                        generated_tokens.append(cand_tok)
                        if (
                            stop_on_eos
                            and tokenizer.eos_token_id is not None
                            and cand_tok == tokenizer.eos_token_id
                        ):
                            break
                    else:
                        diff = torch.clamp(p_probs - q_probs, min=0.0)
                        diff_sum = diff.sum(dim=-1, keepdim=True)
                        if diff_sum.item() > 1e-8:
                            residual_probs = diff / diff_sum
                        else:
                            residual_probs = p_probs
                        final_emitted_token = int(torch.multinomial(residual_probs, num_samples=1).item())
                        break
                bonus_tok = None

            total_accepted += accepted_in_step

            # Check if we stopped early due to EOS inside accepted draft tokens or budget reached
            hit_eos = (
                stop_on_eos
                and tokenizer.eos_token_id is not None
                and len(generated_tokens) > 0
                and generated_tokens[-1] == tokenizer.eos_token_id
            )

            if not hit_eos and len(generated_tokens) < max_new_tokens:
                if accepted_in_step == actual_gamma:
                    # All draft tokens accepted -> sample bonus token from position `actual_gamma`
                    if greedy:
                        final_emitted_token = bonus_tok
                    else:
                        bonus_probs = _compute_probs(
                            verify_logits[:, actual_gamma, :],
                            s_cfg.temperature,
                            s_cfg.top_p,
                            shared_vocab,
                        )
                        final_emitted_token = int(
                            torch.multinomial(bonus_probs, num_samples=1).item()
                        )

                if final_emitted_token is not None:
                    generated_tokens.append(final_emitted_token)
                    if (
                        stop_on_eos
                        and tokenizer.eos_token_id is not None
                        and final_emitted_token == tokenizer.eos_token_id
                    ):
                        hit_eos = True
            verify_time_s += sync_perf_counter(device, s_cfg.timing_mode) - t_v0

            # --- Step 2d: KV-Cache Synchronization & Rollback ---
            t_rb0 = sync_perf_counter(device, s_cfg.timing_mode)
            valid_new_kv_count = accepted_in_step + 1
            cur_target_pos = pre_step_target_len + valid_new_kv_count
            _crop_cache(target_cache, cur_target_pos, use_static_target)

            if accepted_in_step < actual_gamma:
                new_draft_len = pre_step_draft_len + valid_new_kv_count
                _crop_cache(
                    draft_cache, new_draft_len - draft_evicted if use_static_draft else new_draft_len, use_static_draft
                )
                cur_draft_pos = new_draft_len
                if final_emitted_token is not None:
                    last_token = torch.tensor([[final_emitted_token]], dtype=torch.long, device=device)
                    pending_draft_input = last_token.to(draft_device)
            else:
                if final_emitted_token is not None:
                    last_token = torch.tensor([[final_emitted_token]], dtype=torch.long, device=device)
                    if actual_gamma > 0:
                        pending_draft_input = torch.tensor(
                            [[draft_tokens[-1], final_emitted_token]],
                            dtype=torch.long,
                            device=draft_device,
                        )
                    else:
                        pending_draft_input = last_token.to(draft_device)
            rollback_time_s += sync_perf_counter(device, s_cfg.timing_mode) - t_rb0

            # --- Step 2e: Synchronized Adaptive KV-Cache Compression ---
            t_c0 = sync_perf_counter(device, s_cfg.timing_mode)
            if use_shared_keepset:
                # On the last step `pending_draft_input` may be stale; the final sync below handles it.
                last_step = hit_eos or len(generated_tokens) >= max_new_tokens
                if not last_step and (
                    target_cache.should_compress()
                    or (use_shared_adaptive_draft and draft_cache.should_compress())
                ):
                    if pending_draft_input.shape[1] == 2:
                        catchup_tok = pending_draft_input[:, :1]
                        if use_shared_static_draft:
                            _ = _draft_static_step(catchup_tok, cur_draft_pos, s_cfg.compile_draft)
                        else:
                            _ = model_forward_with_logits_to_keep(
                                draft_model,
                                logits_to_keep_count=1 if use_ltk else None,
                                input_ids=catchup_tok,
                                past_key_values=draft_cache,
                                use_cache=True,
                            )
                        cur_draft_pos += 1
                        draft_passes += 1
                        pending_draft_input = pending_draft_input[:, 1:]
                    draft_evicted += _shared_compress(force=True)
            else:
                if not use_static_target and target_cache.should_compress():
                    target_cache.compress_step()
                if not use_static_draft and draft_cache.should_compress():
                    if pending_draft_input.shape[1] == 2:
                        catchup_tok = pending_draft_input[:, :1]
                        _ = model_forward_with_logits_to_keep(
                            draft_model,
                            logits_to_keep_count=1 if use_ltk else None,
                            input_ids=catchup_tok,
                            past_key_values=draft_cache,
                            use_cache=True,
                        )
                        cur_draft_pos += 1
                        draft_passes += 1
                        pending_draft_input = pending_draft_input[:, 1:]
                    draft_cache.compress_step()
            cache_overhead_s += sync_perf_counter(device, s_cfg.timing_mode) - t_c0

            if hit_eos or len(generated_tokens) >= max_new_tokens:
                break

        t_c1 = sync_perf_counter(device, s_cfg.timing_mode)
        if use_shared_keepset:
            # Bring the draft to the target's logical length (it can lag by 1-2 tokens when the run stops on the
            # token budget), from the real token sequence, then apply the final shared compression.
            target_logical = int(target_cache.get_seq_length())
            if cur_draft_pos < target_logical:
                seq_ids = torch.cat(
                    [input_ids[0], torch.tensor(generated_tokens, dtype=torch.long, device=device)]
                ).to(draft_device)
                catchup_ids = seq_ids[cur_draft_pos:target_logical].unsqueeze(0)
                if use_shared_static_draft:
                    _ = _draft_static_step(catchup_ids, cur_draft_pos, False)
                else:
                    _ = model_forward_with_logits_to_keep(
                        draft_model,
                        logits_to_keep_count=1 if use_ltk else None,
                        input_ids=catchup_ids,
                        past_key_values=draft_cache,
                        use_cache=True,
                    )
                cur_draft_pos = target_logical
                draft_passes += 1
            draft_evicted += _shared_compress(force=True)
        else:
            if not use_static_target:
                target_cache.compress_step(force=True)
            if not use_static_draft:
                draft_cache.compress_step(force=True)
        cache_overhead_s += sync_perf_counter(device, s_cfg.timing_mode) - t_c1
        total_time_s = max(1e-6, sync_perf_counter(device, s_cfg.timing_mode or device.type == "cuda") - t_start)
    finally:
        if not use_static_target:
            target_cache.remove_query_hooks()

    peak_vram_mb = get_peak_vram_mb(device)

    if ttft_ms is None:
        ttft_ms = total_time_s * 1000.0

    num_gen = len(generated_tokens)
    decode_time_ms = max(0.0, total_time_s * 1000.0 - ttft_ms)
    tpot_ms = decode_time_ms / max(1, num_gen - 1) if num_gen > 1 else ttft_ms
    tokens_per_sec = num_gen / total_time_s
    decode_tokens_per_sec = (
        ((num_gen - 1) * 1000.0 / decode_time_ms)
        if num_gen > 1 and decode_time_ms > 0
        else tokens_per_sec
    )
    acceptance_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0
    mean_accepted = num_gen / max(1, spec_steps)
    mean_drafted = total_drafted / max(1, spec_steps)
    draft_time_ms = (draft_time_s * 1000.0) / max(1, spec_steps)
    draft_ms_per_token = (draft_time_s * 1000.0) / max(1, total_drafted)
    verify_time_ms = (verify_time_s * 1000.0) / max(1, spec_steps)
    rollback_time_ms = (rollback_time_s * 1000.0) / max(1, spec_steps)
    cache_overhead_ms = (cache_overhead_s * 1000.0) / max(1, spec_steps)

    gen_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
    full_ids = torch.cat([input_ids, gen_tensor], dim=-1)
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return GenerationOutput(
        text=decoded_text,
        output_ids=full_ids,
        num_generated_tokens=num_gen,
        total_time_s=round(total_time_s, 4),
        ttft_ms=round(ttft_ms, 2),
        tpot_ms=round(tpot_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        target_forward_passes=target_passes,
        draft_forward_passes=draft_passes,
        total_drafted_tokens=total_drafted,
        total_accepted_tokens=total_accepted,
        acceptance_rate=round(acceptance_rate, 4),
        mean_accepted_per_step=round(mean_accepted, 2),
        mean_drafted_per_step=round(mean_drafted, 2),
        decode_tokens_per_sec=round(decode_tokens_per_sec, 2),
        draft_time_ms=round(draft_time_ms, 2),
        draft_ms_per_token=round(draft_ms_per_token, 3),
        draft_path=draft_path_label(draft_model, use_static_draft, s_cfg.compile_draft),
        verify_time_ms=round(verify_time_ms, 2),
        rollback_time_ms=round(rollback_time_ms, 3),
        cache_overhead_ms=round(cache_overhead_ms, 3),
        peak_vram_mb=peak_vram_mb,
        cache_stats=(
            static_cache_memory_stats(target_cache, cur_target_pos, device, dtype, "StaticCache")
            if use_static_target
            else target_cache.get_memory_stats()
        ),
        draft_cache_stats=(
            static_cache_memory_stats(
                draft_cache,
                cur_draft_pos,
                draft_device,
                draft_dtype,
                "StaticCache(shared_keepset)" if use_shared_static_draft else "StaticCache",
                phys_len=(cur_draft_pos - draft_evicted) if use_shared_static_draft else None,
            )
            if use_static_draft
            else draft_cache.get_memory_stats()
        ),
        draft_diag=draft_diag,
        draft_cache_mode=(
            "shared_keepset" if use_shared_keepset else ("full" if not t_cache_cfg.enabled else "full(adaptive_target)")
        ),
    )


@torch.no_grad()
def profile_speculative_step(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    spec_config: Optional[SpeculativeConfig] = None,
    target_cache_config: Optional[AdaptiveCacheConfig] = None,
    top_n_ops: int = 10,
) -> Dict[str, Any]:
    """
    Profiles speculative decode steps using `torch.profiler` to surface which CPU/CUDA
    operators scale with context length (e.g. attention vs matmul vs cat/copy_).
    """
    from torch.profiler import ProfilerActivity, profile

    device = next(target_model.parameters()).device
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup short generation first
    _ = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompt=prompt[:128],
        max_new_tokens=8,
        spec_config=spec_config,
        target_cache_config=target_cache_config,
        stop_on_eos=False,
    )

    with profile(activities=activities, record_shapes=False) as prof:
        out = speculative_generate(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=16,
            spec_config=spec_config,
            target_cache_config=target_cache_config,
            stop_on_eos=False,
        )

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    events = sorted(
        prof.key_averages(),
        key=lambda evt: getattr(evt, sort_key, evt.cpu_time_total),
        reverse=True,
    )[:top_n_ops]

    top_ops = []
    for evt in events:
        cuda_us = getattr(evt, "self_cuda_time_total", 0.0)
        top_ops.append(
            {
                "name": evt.key,
                "calls": int(evt.count),
                "self_cpu_ms": round(evt.self_cpu_time_total / 1000.0, 3),
                "self_cuda_ms": round(cuda_us / 1000.0, 3),
            }
        )

    return {
        "prompt_tokens": int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]),
        "draft_time_ms": out.draft_time_ms,
        "verify_time_ms": out.verify_time_ms,
        "rollback_time_ms": out.rollback_time_ms,
        "cache_overhead_ms": out.cache_overhead_ms,
        "top_ops": top_ops,
    }

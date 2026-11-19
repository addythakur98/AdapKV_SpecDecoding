"""
Shared execution utilities for Steps 1, 4, 5, and 6:
  - CUDA-synchronized timing helper (`--timing_mode`)
  - `StaticCache` 8-aligned sizing, O(1) rollback (`crop`), and memory telemetry
  - Bucketed `torch.compile(mode="reduce-overhead")` CUDA-graph runner with safe CPU/Inductor fallback
  - `logits_to_keep` forward support detection & peak GPU VRAM tracking
"""

import inspect
import time
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel, StaticCache


class ContextLengthOverflowError(RuntimeError):
    """Raised when a sequence exceeds a model's max_position_embeddings under strict draft cache consistency."""


def round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """Rounds `value` up to the nearest positive multiple (e.g., 8 for Tensor Core / SDPA alignment)."""
    if value <= 0:
        return multiple
    return ((int(value) + multiple - 1) // multiple) * multiple


def sync_perf_counter(device: torch.device, timing_mode: bool) -> float:
    """
    Step 1: Synchronizes the CUDA device before reading `time.perf_counter()` when `timing_mode=True`
    so asynchronous GPU work from an earlier phase is never billed to the next timer window.
    """
    if timing_mode and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def reset_peak_vram(device: torch.device) -> None:
    """Resets CUDA peak memory stats if executing on GPU."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram_mb(device: torch.device) -> float:
    """Returns `torch.cuda.max_memory_allocated(device)` in MiB (or 0.0 on CPU)."""
    if device.type == "cuda" and torch.cuda.is_available():
        return round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2), 2)
    return 0.0


_LOGITS_TO_KEEP_SUPPORT: Dict[int, bool] = {}


def supports_logits_to_keep(model: PreTrainedModel) -> bool:
    """Checks once whether `model.forward` accepts `logits_to_keep`."""
    mid = id(model.__class__)
    if mid not in _LOGITS_TO_KEEP_SUPPORT:
        try:
            sig = inspect.signature(model.forward)
            _LOGITS_TO_KEEP_SUPPORT[mid] = "logits_to_keep" in sig.parameters
        except Exception:
            _LOGITS_TO_KEEP_SUPPORT[mid] = False
    return _LOGITS_TO_KEEP_SUPPORT[mid]


def model_forward_with_logits_to_keep(
    model: PreTrainedModel,
    logits_to_keep_count: Optional[int] = None,
    **kwargs: Any,
):
    """Invokes `model(**kwargs)` passing `logits_to_keep` when requested and supported."""
    if logits_to_keep_count is not None and logits_to_keep_count > 0 and supports_logits_to_keep(model):
        return model(**kwargs, logits_to_keep=int(logits_to_keep_count))
    return model(**kwargs)


def crop_static_cache(cache: StaticCache, max_length: int) -> None:
    """Resets initialized `StaticCache` layer cursors in O(1) without tensor copies."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for layer in layers:
        if getattr(layer, "is_initialized", False) and hasattr(layer, "cumulative_length"):
            layer.cumulative_length.fill_(int(max_length))


@torch.no_grad()
def compact_static_cache(cache: StaticCache, keep_idx: torch.Tensor) -> int:
    """
    Shared keep-set eviction for a StaticCache, in place (static addresses / CUDA graphs stay valid):
    gathers slots `keep_idx` (sorted ascending, all < current length) of every layer into the front of the
    buffer and sets the physical length (`cumulative_length`) to len(keep_idx). Keys keep the RoPE rotation of
    their original (logical) positions. Slots past the new length are stale and masked out. Returns new length.
    """
    kept = int(keep_idx.numel())
    for layer in getattr(cache, "layers", []) or []:
        if not getattr(layer, "is_initialized", False):
            continue
        idx = keep_idx.to(layer.keys.device)
        layer.keys[:, :, :kept, :].copy_(layer.keys.index_select(2, idx))
        layer.values[:, :, :kept, :].copy_(layer.values.index_select(2, idx))
        layer.cumulative_length.fill_(kept)
    return kept


def static_cache_is_static_address(cache: StaticCache) -> Optional[bool]:
    """True if StaticCache K/V are marked via torch._dynamo.mark_static_address (CUDA-graph replay won't copy them)."""
    layers = [l for l in getattr(cache, "layers", []) or [] if getattr(l, "is_initialized", False)]
    if not layers:
        return None
    return all(
        getattr(l.keys, "_dynamo_static_input_type", None) is not None
        and getattr(l.values, "_dynamo_static_input_type", None) is not None
        for l in layers
    )


def static_cache_memory_stats(
    cache: StaticCache,
    seen: int,
    device: torch.device,
    dtype: torch.dtype,
    label: str = "StaticCache",
    phys_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Reports allocated vs logical KV-cache footprint for a `StaticCache`. `seen` is the logical length;
    `phys_len` (shared keep-set draft) is the number of slots actually holding kept tokens.
    """
    layers = getattr(cache, "layers", [])
    actual_bytes = 0
    num_layers = len(layers)
    sample_shape = None
    sample_elem = torch.tensor([], dtype=dtype).element_size()
    for layer in layers:
        k_t = getattr(layer, "keys", None)
        v_t = getattr(layer, "values", None)
        if k_t is not None and v_t is not None:
            actual_bytes += k_t.numel() * k_t.element_size() + v_t.numel() * v_t.element_size()
            if sample_shape is None:
                sample_shape = k_t.shape
                sample_elem = k_t.element_size()
    max_cache_len = int(sample_shape[2]) if sample_shape is not None else 0
    if sample_shape is not None:
        b, h, _, d = sample_shape
        uncomp_bytes = num_layers * 2 * b * h * int(seen) * d * sample_elem
    else:
        uncomp_bytes = actual_bytes
    if phys_len is not None and sample_shape is not None:
        kept = int(phys_len)
        used_bytes = num_layers * 2 * b * h * kept * d * sample_elem
        return {
            "num_layers": num_layers,
            "active_seq_len": kept,
            "fp_tokens": kept,
            "quantized_tokens": 0,
            "pruned_tokens": max(0, int(seen) - kept),
            "total_tokens_seen": int(seen),
            "max_cache_len": max_cache_len,
            # actual = kept tokens (same accounting as the target AdaptiveKVCache); allocated = whole buffer
            "actual_memory_mb": round(used_bytes / (1024.0 ** 2), 4),
            "allocated_memory_mb": round(actual_bytes / (1024.0 ** 2), 4),
            "uncompressed_memory_mb": round(uncomp_bytes / (1024.0 ** 2), 4),
            "compression_ratio": round(uncomp_bytes / max(1, used_bytes), 2) if used_bytes > 0 else 1.0,
            "token_compression_ratio": round(int(seen) / max(1, kept), 2),
            "token_retention_pct": round(100.0 * kept / max(1, int(seen)), 1),
            "memory_saved_pct": round(100.0 * (1.0 - used_bytes / max(1, uncomp_bytes)), 2),
            "cache_type": label,
        }
    return {
        "num_layers": num_layers,
        "active_seq_len": int(seen),
        "fp_tokens": int(seen),
        "quantized_tokens": 0,
        "pruned_tokens": 0,
        "total_tokens_seen": int(seen),
        "max_cache_len": max_cache_len,
        "actual_memory_mb": round(actual_bytes / (1024.0 ** 2), 4),
        "uncompressed_memory_mb": round(uncomp_bytes / (1024.0 ** 2), 4),
        "compression_ratio": round(uncomp_bytes / max(1, actual_bytes), 2) if actual_bytes > 0 else 1.0,
        "token_compression_ratio": 1.0,
        "token_retention_pct": 100.0,
        "memory_saved_pct": 0.0,
        "cache_type": label,
    }


# Cached compiled step runners and pooled StaticCaches (reuses CUDA graph static addresses across runs)
_COMPILED_RUNNER_CACHE: Dict[Tuple[Any, ...], Callable[..., torch.Tensor]] = {}
_PERSISTENT_STATIC_CACHES: Dict[Tuple[int, int, int], StaticCache] = {}
_COMPILE_DISABLED_MODELS: Dict[int, str] = {}
# Per-model compile / CUDA-graph status for reporting (keyed by id(model))
_COMPILE_STATUS: Dict[int, Dict[str, Any]] = {}
_DYNAMO_LIMITS_RAISED = False


def _raise_dynamo_cache_limits() -> None:
    """
    Dynamo's default recompile limit (8 per code object) is shared by every model / StaticCache object that
    goes through `_step`. A sweep over lengths x configs exceeds it, after which dynamo silently runs eager.
    Raise it once so each (model, cache bucket) gets its own compiled graph.
    """
    global _DYNAMO_LIMITS_RAISED
    if _DYNAMO_LIMITS_RAISED:
        return
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = max(int(getattr(_dynamo.config, "cache_size_limit", 8)), 64)
        if hasattr(_dynamo.config, "accumulated_cache_size_limit"):
            _dynamo.config.accumulated_cache_size_limit = max(
                int(_dynamo.config.accumulated_cache_size_limit), 512
            )
    except Exception:
        pass
    _DYNAMO_LIMITS_RAISED = True


def _set_compile_status(model: PreTrainedModel, **fields: Any) -> None:
    st = _COMPILE_STATUS.setdefault(id(model), {"state": "eager", "mode": None, "reason": "", "graphs": 0})
    st.update(fields)


def get_compile_status(model: Optional[PreTrainedModel]) -> Dict[str, Any]:
    """Returns {'state': 'cudagraphs'|'compiled'|'eager'|'eager_fallback', 'mode', 'reason', 'graphs'} for `model`."""
    if model is None:
        return {"state": "none", "mode": None, "reason": "", "graphs": 0}
    return dict(_COMPILE_STATUS.get(id(model), {"state": "eager", "mode": None, "reason": "", "graphs": 0}))


def dynamo_counters_summary() -> Dict[str, Any]:
    """Graph breaks, unique graphs and cudagraph skips recorded by torch._dynamo / inductor in this process."""
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    graph_breaks = {str(k): int(v) for k, v in counters.get("graph_break", {}).items()}
    return {
        "unique_graphs": int(counters.get("stats", {}).get("unique_graphs", 0)),
        "frames_compiled_ok": int(counters.get("frames", {}).get("ok", 0)),
        "graph_breaks_total": int(sum(graph_breaks.values())),
        "graph_break_reasons": graph_breaks,
        "recompiles": int(counters.get("recompiles", {}).get("total", 0)) if "recompiles" in counters else None,
        "cudagraph_skips": int(counters.get("inductor", {}).get("cudagraph_skips", 0)),
    }


def draft_path_label(model: Optional[PreTrainedModel], use_static: bool, compile_requested: bool) -> str:
    """Short human-readable draft execution path, e.g. 'sdpa_gqa_decode|static|cudagraphs'."""
    if model is None:
        return "none"
    attn = str(getattr(model.config, "_attn_implementation", "?"))
    if not use_static:
        return f"{attn}|dynamic|eager"
    if not compile_requested:
        return f"{attn}|static|eager"
    st = get_compile_status(model)
    state = st.get("state", "eager")
    if state == "eager":
        state = "eager(compile not used)"
    return f"{attn}|static|{state}"


def static_cache_bucket(needed: int, max_positions: int, round_to_8: bool = True) -> int:
    """
    Capacity bucket for a pooled draft StaticCache: next power of two >= `needed` (min 1024) up to 4096, then
    the next multiple of 2048, capped at `max_positions`. Keeps one StaticCache (and one CUDA graph) per bucket
    instead of one per document, so per-doc prompt-length differences rarely trigger a recompile. The draft's
    decode attention reads the whole capacity (masked), so buckets are kept tight (a pure power-of-two bucket
    made a 16K context read a 32K buffer). Always a multiple of 8.
    """
    cap = 1024
    while cap < needed and cap < 4096:
        cap *= 2
    if cap < needed:
        cap = ((int(needed) + 2047) // 2048) * 2048
    if cap > max_positions:
        cap = max(int(needed), int(max_positions))
    return round_up_to_multiple(cap, 8) if round_to_8 else cap


def clear_compiled_runners() -> None:
    """Clears cached compiled runners and pooled StaticCaches to reclaim VRAM."""
    _COMPILED_RUNNER_CACHE.clear()
    _PERSISTENT_STATIC_CACHES.clear()
    _COMPILE_DISABLED_MODELS.clear()


def release_model_static_caches(model: Optional[PreTrainedModel] = None) -> None:
    """Drops pooled StaticCaches (all or for a specific model) so VRAM can be freed on OOM or length change."""
    if model is None:
        clear_compiled_runners()
        return
    mid = id(model)
    for k in [k for k in _PERSISTENT_STATIC_CACHES if k[0] == mid]:
        _PERSISTENT_STATIC_CACHES.pop(k, None)
    for k in [k for k in _COMPILED_RUNNER_CACHE if k[0] == mid]:
        _COMPILED_RUNNER_CACHE.pop(k, None)


def get_reusable_static_cache(
    model: PreTrainedModel,
    max_cache_len: int,
    batch_size: int = 1,
) -> StaticCache:
    """
    Returns a pooled `StaticCache` for `(model, max_cache_len, batch_size)` with `cumulative_length`
    reset to 0 so `torch.compile(mode='reduce-overhead')` keeps static tensor addresses across docs/repeats
    without leaking old KV buffers across context lengths.
    """
    mid = id(model)
    key = (mid, int(max_cache_len), int(batch_size))
    existing = _PERSISTENT_STATIC_CACHES.get(key)
    if existing is not None:
        crop_static_cache(existing, 0)
        return existing

    # Release other StaticCache capacities for the same model so multiple 8k/16k pools don't accumulate
    for old_k in [k for k in _PERSISTENT_STATIC_CACHES if k[0] == mid and k != key]:
        old_cache = _PERSISTENT_STATIC_CACHES.pop(old_k, None)
        if old_cache is not None:
            old_id = id(old_cache)
            for r_k in [r_k for r_k in _COMPILED_RUNNER_CACHE if r_k[1] == old_id]:
                _COMPILED_RUNNER_CACHE.pop(r_k, None)

    cache = StaticCache(config=model.config, max_cache_len=int(max_cache_len))
    _PERSISTENT_STATIC_CACHES[key] = cache
    return cache


def _make_static_step_fn(
    model: PreTrainedModel,
    past_key_values: Any,
    logits_to_keep_count: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    use_ltk = logits_to_keep_count > 0 and supports_logits_to_keep(model)

    def _step(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        if use_ltk:
            out = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=logits_to_keep_count,
            )
        else:
            out = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return out.logits[:, -logits_to_keep_count:, :] if logits_to_keep_count > 0 else out.logits

    return _step


def run_bucketed_static_step(
    model: PreTrainedModel,
    past_key_values: StaticCache,
    input_ids: torch.Tensor,
    cur_seen: int,
    logits_to_keep_count: int = 1,
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    logical_pos: Optional[int] = None,
) -> torch.Tensor:
    """
    `cur_seen` is the PHYSICAL write slot (StaticCache cumulative_length; mask length). `logical_pos` is the
    token's true absolute position used for RoPE `position_ids` (defaults to `cur_seen`). They differ only for
    a compacted (shared keep-set) draft cache, where evicted tokens leave the buffer but keep their positions.

    Step 4: Executes one decode (`q_len=1`) or verify (`q_len=k`) step against a `StaticCache`,
    using a bucketed `torch.compile(mode=compile_mode)` runner keyed by `(q_len, max_cache_len)` when enabled.

    Handles flash_attention_2 + StaticCache safety:
    - Under compiled static graphs: switches `_attn_implementation` to `'sdpa'` during the static step
      so fixed-shape `(B, 1, q_len, max_cache_len)` causal masking works without `.item()` graph breaks
      or reading zeroed future StaticCache slots, then restores the previous `_attn_implementation`.
    - Under uncompiled FA2: supplies an explicit boolean `attention_mask` of length `cur_seen + q_len`
      so FA2 `_upad_input` slices valid keys only.
    """
    device = input_ids.device
    batch_size, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    pos = torch.arange(cur_seen, cur_seen + q_len, device=device, dtype=torch.long)
    cache_position = pos
    if logical_pos is None or int(logical_pos) == int(cur_seen):
        position_ids = pos.unsqueeze(0).expand(batch_size, -1)
    else:
        position_ids = (
            torch.arange(int(logical_pos), int(logical_pos) + q_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
    max_cap_check = int(getattr(past_key_values.layers[0], "max_cache_len", 0) or 0)
    if max_cap_check and cur_seen + q_len > max_cap_check:
        raise RuntimeError(
            f"StaticCache overflow: writing slots [{cur_seen}, {cur_seen + q_len}) into capacity {max_cap_check}."
        )

    max_cache_len = int(
        getattr(
            past_key_values.layers[0],
            "max_cache_len",
            getattr(past_key_values, "max_cache_len", 0),
        )
    )

    prev_attn = getattr(model.config, "_attn_implementation", None)

    if compile_enabled and id(model) not in _COMPILE_DISABLED_MODELS and hasattr(torch, "compile"):
        # CUDA graphs (`reduce-overhead`) require CUDA or fallback cleanly on CPU without cl/triton
        use_mode = compile_mode if device.type == "cuda" else "default"
        key = (
            id(model),
            id(past_key_values),
            q_len,
            max_cache_len,
            logits_to_keep_count,
            use_mode,
        )
        runner = _COMPILED_RUNNER_CACHE.get(key)
        if runner is None:
            _raise_dynamo_cache_limits()
            raw_fn = _make_static_step_fn(model, past_key_values, logits_to_keep_count)
            try:
                if device.type == "cuda":
                    runner = torch.compile(raw_fn, mode=use_mode, fullgraph=False, dynamic=False)
                else:
                    # On CPU without C++ compiler/Triton, test dynamo tracing via aot_eager only when safe,
                    # otherwise compile normally and catch InductorError on first call.
                    runner = torch.compile(raw_fn, mode=use_mode, fullgraph=False, dynamic=False)
                if len(_COMPILED_RUNNER_CACHE) > 64:
                    _COMPILED_RUNNER_CACHE.clear()
                _COMPILED_RUNNER_CACHE[key] = runner
            except Exception as exc:
                _COMPILE_DISABLED_MODELS[id(model)] = str(exc)
                _set_compile_status(model, state="eager_fallback", mode=use_mode, reason=f"{type(exc).__name__}: {exc}"[:160])
                print(f"[CompileRunner] Note: torch.compile unavailable ({exc}); using eager execution.")
                runner = None

        if runner is not None:
            try:
                if prev_attn == "flash_attention_2":
                    model.config._attn_implementation = "sdpa"
                # Fresh contiguous inputs: a (1,1) slice of the prompt keeps the prompt-length stride, which
                # fails dynamo's stride guard and recompiles (and re-captures the CUDA graph) for every document.
                result = runner(
                    input_ids.clone(memory_format=torch.contiguous_format),
                    position_ids.clone(memory_format=torch.contiguous_format),
                    cache_position.clone(memory_format=torch.contiguous_format),
                )
                graphs = len([k for k in _COMPILED_RUNNER_CACHE if k[0] == id(model)])
                _set_compile_status(
                    model,
                    state="cudagraphs" if (device.type == "cuda" and use_mode == "reduce-overhead") else "compiled",
                    mode=use_mode,
                    reason="",
                    graphs=graphs,
                )
                return result
            except Exception as exc:
                _COMPILE_DISABLED_MODELS[id(model)] = str(exc)
                _set_compile_status(model, state="eager_fallback", mode=use_mode, reason=f"{type(exc).__name__}: {exc}"[:160])
                _COMPILED_RUNNER_CACHE.pop(key, None)
                print(
                    f"[CompileRunner] Note: compiled step fallback to eager for q_len={q_len} "
                    f"({type(exc).__name__}: {exc})."
                )
            finally:
                if prev_attn == "flash_attention_2":
                    model.config._attn_implementation = prev_attn

    # Eager StaticCache path
    if prev_attn == "flash_attention_2":
        # Uncompiled FA2 needs explicit 2D mask up to valid `cur_seen + q_len` so future StaticCache zeros aren't read
        attn_mask = torch.ones((batch_size, cur_seen + q_len), dtype=torch.bool, device=device)
        out = model_forward_with_logits_to_keep(
            model,
            logits_to_keep_count=logits_to_keep_count,
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
    else:
        out = model_forward_with_logits_to_keep(
            model,
            logits_to_keep_count=logits_to_keep_count,
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return out.logits[:, -logits_to_keep_count:, :] if logits_to_keep_count > 0 else out.logits

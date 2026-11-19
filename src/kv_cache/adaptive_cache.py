"""
Adaptive KV-Cache Compression (Dynamic Token Pruning + Mixed-Precision Quantization).

Implements a high-throughput, two-pronged KV cache manager compatible with Hugging Face `DynamicCache`:
1. Zero-Overhead Hot-Path `update()`:
   - During standard forward passes, `update()` performs only a fast tensor append on the
     active Full-Precision Hot Tier (`[Attention Sinks | Heavy-Hitters | Recent Window]`).
   - Relevance scoring, pruning, and INT4 bit-packing are amortized and executed only inside
     `compress_step()` once the buffer exceeds `max_cache_size` by a configurable stride.
2. Three-Tier Memory Hierarchy:
   - Hot Full-Precision Tier (FP16/BF16): Attention Sinks + Top Heavy-Hitters + Recent Window
     (actively attended to during SDPA, reducing attention FLOPs and memory bandwidth).
   - Warm Quantized Archive Tier (Packed INT4/INT8): Medium-importance historical tokens
     compressed 4x in memory (`quant_key_blocks` / `quant_value_blocks`).
   - Cold Evicted Tier: Low-importance tokens pruned from memory.
3. Speculative Rollback Support: `crop(max_length)` cleanly rolls back rejected draft
   tokens in O(1) time during speculative decoding before compression is committed.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers.cache_utils import DynamicCache

from config import AdaptiveCacheConfig
from src.kv_cache.quantizer import (
    QuantizedKVBlock,
    concat_quantized_blocks,
    dequantize_tensor,
    quantize_tensor,
)


@dataclass
class LayerCacheStats:
    """Memory and token distribution statistics for a single transformer layer."""

    fp_tokens: int
    quant_tokens: int
    pruned_tokens: int
    actual_bytes: int
    uncompressed_bytes: int


class AdaptiveKVCache(DynamicCache):
    """
    Adaptive KV-Cache supporting dynamic pruning, mixed-precision INT4/INT8 quantization
    (actively dequantized into attention), monotonic RoPE position tracking, and
    speculative decoding rollback (`crop`).
    """

    def __init__(self, config: Optional[AdaptiveCacheConfig] = None) -> None:
        super().__init__()
        self.config = config or AdaptiveCacheConfig()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0

        # Preallocated contiguous backing buffers for O(1) in-place update & rollback (avoids full-seq torch.cat)
        self._k_buf: List[Optional[torch.Tensor]] = []
        self._v_buf: List[Optional[torch.Tensor]] = []
        self._fp_len: List[int] = []

        # Per-layer Warm Tier: packed INT4/INT8 quantized blocks (kept quantized in VRAM)
        self.quant_key_blocks: List[Optional[QuantizedKVBlock]] = []
        self.quant_value_blocks: List[Optional[QuantizedKVBlock]] = []

        # Cached dequantized Warm Tier tensors (invalidated when Warm blocks or tensor versions change)
        self._warm_k_dequant: List[Optional[torch.Tensor]] = []
        self._warm_v_dequant: List[Optional[torch.Tensor]] = []
        self._warm_sig: List[Optional[Tuple[Any, ...]]] = []

        # Per-layer observation window inputs captured via lightweight pre-hooks for Q_obs @ K^T scoring
        self._obs_inputs: List[Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Any]]] = []
        self._hook_handles: List[Any] = []

        # Per-layer cumulative importance scores for currently held middle FP tokens
        self.importance_scores: List[Optional[torch.Tensor]] = []

        # Total logical tokens ever inserted per layer (for RoPE positions & compression ratio)
        self.total_tokens_seen_per_layer: List[int] = []
        self.pruned_tokens_per_layer: List[int] = []
        self._last_layer_mid_scores: List[Optional[torch.Tensor]] = []
        self._last_keep_indices: List[Optional[torch.Tensor]] = []
        # Shared keep-set bookkeeping: incremented whenever compress_step() actually evicts; layer-0 geometry
        # (fp_len, actual_sink, mid_end, keep_hh) of that compression, used to build the draft's keep-set.
        self.compress_events: int = 0
        self._last_compress_geom: Optional[Tuple[int, int, int, int]] = None

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def _ensure_layer_initialized(self, layer_idx: int) -> None:
        while len(self.quant_key_blocks) <= layer_idx:
            self._k_buf.append(None)
            self._v_buf.append(None)
            self._fp_len.append(0)
            self.quant_key_blocks.append(None)
            self.quant_value_blocks.append(None)
            self._warm_k_dequant.append(None)
            self._warm_v_dequant.append(None)
            self._warm_sig.append(None)
            self._obs_inputs.append(None)
            self.importance_scores.append(None)
            self.total_tokens_seen_per_layer.append(0)
            self.pruned_tokens_per_layer.append(0)
            self._last_layer_mid_scores.append(None)
            self._last_keep_indices.append(None)

    def register_query_hooks(self, model: torch.nn.Module) -> None:
        """
        Attaches zero-matmul forward pre-hooks on transformer `self_attn` modules to capture
        the trailing observation window `(hidden_states, cos, sin)` for true SnapKV-style
        `Q_obs @ K^T` heavy-hitter scoring during `compress_step()`.
        """
        self.remove_query_hooks()
        if not self.config.enabled:
            return

        obs_w = max(1, int(getattr(self.config, "obs_window_size", 16)))
        attn_modules: List[Tuple[int, torch.nn.Module]] = []
        for mod in model.modules():
            sa = getattr(mod, "self_attn", None)
            if sa is not None and hasattr(sa, "q_proj"):
                attn_modules.append((len(attn_modules), sa))

        for layer_idx, sa in attn_modules:
            self._ensure_layer_initialized(layer_idx)

            def _make_pre_hook(l_idx: int, attn_mod: torch.nn.Module):
                def _pre_hook(_mod, args, kwargs):
                    if not self.config.enabled:
                        return
                    if kwargs.get("past_key_values") is not self:
                        return
                    hs = kwargs.get("hidden_states")
                    if hs is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                        hs = args[0]
                    if hs is None or hs.ndim != 3:
                        return
                    q_len = hs.shape[1]

                    # Skip recording during decode steps far away from the next compression boundary
                    if q_len < obs_w and l_idx < len(self._fp_len) and self._fp_len[l_idx] > 0:
                        seen_0 = self.get_seq_length(0)
                        _, _, _, fp_budget, amortize_stride = self._effective_budgets(seen_0)
                        trigger = fp_budget + amortize_stride
                        if self._fp_len[l_idx] + q_len < trigger - obs_w:
                            return

                    pos_emb = kwargs.get("position_embeddings")
                    cos_t = pos_emb[0] if (pos_emb is not None and len(pos_emb) == 2) else None
                    sin_t = pos_emb[1] if (pos_emb is not None and len(pos_emb) == 2) else None

                    if q_len >= obs_w:
                        h_slice = hs[:, -obs_w:, :].detach()
                        c_slice = cos_t[:, -obs_w:, :].detach() if cos_t is not None else None
                        s_slice = sin_t[:, -obs_w:, :].detach() if sin_t is not None else None
                        self._obs_inputs[l_idx] = (h_slice, c_slice, s_slice, attn_mod)
                    else:
                        prev = self._obs_inputs[l_idx]
                        h_new = hs.detach()
                        c_new = cos_t.detach() if cos_t is not None else None
                        s_new = sin_t.detach() if sin_t is not None else None
                        if prev is not None and prev[0].shape[0] == h_new.shape[0] and prev[0].device == h_new.device:
                            h_cat = torch.cat([prev[0], h_new], dim=1)[:, -obs_w:, :]
                            c_cat = (
                                torch.cat([prev[1], c_new], dim=1)[:, -obs_w:, :]
                                if (prev[1] is not None and c_new is not None)
                                else c_new
                            )
                            s_cat = (
                                torch.cat([prev[2], s_new], dim=1)[:, -obs_w:, :]
                                if (prev[2] is not None and s_new is not None)
                                else s_new
                            )
                            self._obs_inputs[l_idx] = (h_cat, c_cat, s_cat, attn_mod)
                        else:
                            self._obs_inputs[l_idx] = (h_new, c_new, s_new, attn_mod)

                return _pre_hook

            handle = sa.register_forward_pre_hook(_make_pre_hook(layer_idx, sa), with_kwargs=True)
            self._hook_handles.append(handle)

    def remove_query_hooks(self) -> None:
        """Detaches any active observation-window query hooks."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _effective_budgets(self, seen_tokens: int) -> Tuple[int, int, int, int, int]:
        """
        Computes (effective_max, hh_budget, quant_budget, fp_budget, amortize_stride).
        Supports proportional retention (`cache_budget_ratio`) on long 4K-32K prompts.
        """
        base_max = self.config.max_cache_size
        if self.config.cache_budget_ratio is not None and self.config.cache_budget_ratio > 0.0:
            scaled = int(round(seen_tokens * self.config.cache_budget_ratio))
            effective_max = max(base_max, scaled)
            if self.config.max_cache_ceiling is not None:
                effective_max = min(effective_max, self.config.max_cache_ceiling)
        else:
            effective_max = base_max

        sink_size = self.config.num_sink_tokens
        recent_size = self.config.recent_window_size
        mid_budget_total = max(0, effective_max - sink_size - recent_size)
        hh_budget = int(round(mid_budget_total * self.config.heavy_hitter_ratio))
        quant_budget = mid_budget_total - hh_budget
        fp_budget = sink_size + hh_budget + recent_size
        amortize_stride = min(512, max(48, mid_budget_total // 4 if mid_budget_total > 0 else 48))
        return effective_max, hh_budget, quant_budget, fp_budget, amortize_stride

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new KV states to the Hot FP Tier and returns the combined
        [Warm Quantized (dequantized) | Hot FP] Key/Value states for attention.
        """
        if len(self.quant_key_blocks) <= layer_idx:
            self._ensure_layer_initialized(layer_idx)

        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += new_tokens
        self.total_tokens_seen_per_layer[layer_idx] += new_tokens

        while len(self.key_cache) <= layer_idx:
            empty_k = torch.empty(
                (key_states.shape[0], key_states.shape[1], 0, key_states.shape[3]),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            self.key_cache.append(empty_k)
            self.value_cache.append(empty_k)

        cur_len = self._fp_len[layer_idx]
        needed = cur_len + new_tokens
        buf_k = self._k_buf[layer_idx]
        buf_v = self._v_buf[layer_idx]

        if buf_k is None or buf_k.shape[-2] < needed or buf_k.device != key_states.device:
            slack = min(256, max(64, needed // 8))
            new_cap = needed + slack
            b, h, _, d = key_states.shape
            new_buf_k = torch.empty((b, h, new_cap, d), dtype=key_states.dtype, device=key_states.device)
            new_buf_v = torch.empty((b, h, new_cap, d), dtype=value_states.dtype, device=value_states.device)
            if cur_len > 0 and buf_k is not None:
                new_buf_k[:, :, :cur_len, :].copy_(buf_k[:, :, :cur_len, :])
                new_buf_v[:, :, :cur_len, :].copy_(buf_v[:, :, :cur_len, :])
            self._k_buf[layer_idx] = new_buf_k
            self._v_buf[layer_idx] = new_buf_v
            buf_k = new_buf_k
            buf_v = new_buf_v

        buf_k[:, :, cur_len:needed, :].copy_(key_states)
        buf_v[:, :, cur_len:needed, :].copy_(value_states)
        self._fp_len[layer_idx] = needed

        hot_k = buf_k[:, :, :needed, :]
        hot_v = buf_v[:, :, :needed, :]
        self.key_cache[layer_idx] = hot_k
        self.value_cache[layer_idx] = hot_v
        q_k = self.quant_key_blocks[layer_idx]
        q_v = self.quant_value_blocks[layer_idx]

        if q_k is not None and q_k.seq_length > 0 and q_v is not None and q_v.seq_length > 0:
            sig_k = (
                tuple((id(s), int(s.q_data._version), int(s.scale._version)) for s in q_k.segments)
                if q_k.segments is not None
                else (id(q_k), int(q_k.q_data._version), int(q_k.scale._version))
            )
            sig_v = (
                tuple((id(s), int(s.q_data._version), int(s.scale._version)) for s in q_v.segments)
                if q_v.segments is not None
                else (id(q_v), int(q_v.q_data._version), int(q_v.scale._version))
            )
            cur_sig = (sig_k, sig_v)
            warm_k = self._warm_k_dequant[layer_idx]
            warm_v = self._warm_v_dequant[layer_idx]
            if warm_k is None or warm_v is None or self._warm_sig[layer_idx] != cur_sig:
                warm_k = dequantize_tensor(q_k)
                warm_v = dequantize_tensor(q_v)
                self._warm_k_dequant[layer_idx] = warm_k
                self._warm_v_dequant[layer_idx] = warm_v
                self._warm_sig[layer_idx] = cur_sig
            return torch.cat([warm_k, hot_k], dim=-2), torch.cat([warm_v, hot_v], dim=-2)

        return hot_k, hot_v

    @torch.no_grad()
    def _compute_token_relevance(
        self,
        layer_idx: int,
        mid_k: torch.Tensor,
        mid_v: torch.Tensor,
        recent_k: torch.Tensor,
        fp_k: Optional[torch.Tensor] = None,
        actual_sink: int = 0,
        mid_end: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes composite token relevance scores for candidate middle tokens.
        When observation queries were captured via `register_query_hooks(model)`, computes true
        SnapKV-style `Q_obs @ K^T` global causal attention probabilities over `[sink | mid | recent]`.
        Falls back cleanly to recent-key probing when used standalone without hooks.
        """
        mid_len = mid_k.shape[-2]
        head_dim = mid_k.shape[-1]

        obs = self._obs_inputs[layer_idx] if layer_idx < len(self._obs_inputs) else None
        used_query_scoring = False
        if obs is not None:
            h_obs, cos_obs, sin_obs, attn_mod = obs
            try:
                bsz, obs_len, _ = h_obs.shape
                num_kv_heads = mid_k.shape[1]
                q_proj_out = attn_mod.q_proj(h_obs)
                num_q_heads = q_proj_out.shape[-1] // head_dim
                num_kv_groups = max(1, num_q_heads // max(1, num_kv_heads))

                q_obs = q_proj_out.view(bsz, obs_len, num_q_heads, head_dim).transpose(1, 2)
                if hasattr(attn_mod, "q_norm") and attn_mod.q_norm is not None:
                    q_obs = attn_mod.q_norm(q_obs)
                if cos_obs is not None and sin_obs is not None:
                    c_u = cos_obs.unsqueeze(1)
                    s_u = sin_obs.unsqueeze(1)
                    q_obs = (q_obs * c_u) + (self._rotate_half(q_obs) * s_u)

                # Group queries by KV head so we never allocate repeat_kv copies of full K
                q_grouped = q_obs.reshape(bsz, num_kv_heads, num_kv_groups * obs_len, head_dim).float()
                k_target = fp_k if (fp_k is not None and mid_end is not None) else mid_k
                logits = torch.matmul(q_grouped, k_target.float().transpose(-1, -2)) / (head_dim ** 0.5)

                if k_target is fp_k and obs_len > 1 and logits.shape[-1] >= obs_len:
                    # Mask future tokens inside trailing obs_len columns (causal alignment per group)
                    tri = torch.triu(
                        torch.ones((obs_len, obs_len), device=logits.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    tri_grouped = tri.repeat(num_kv_groups, 1).view(1, 1, num_kv_groups * obs_len, obs_len)
                    logits[..., -obs_len:] = logits[..., -obs_len:].masked_fill(tri_grouped, float("-inf"))

                full_probs = torch.softmax(logits, dim=-1)
                mid_probs = full_probs[..., actual_sink:mid_end] if k_target is fp_k else full_probs
                # Reshape back -> (B, num_kv_heads, num_kv_groups, obs_len, mid_len) -> pool over groups & obs
                per_head_mid = mid_probs.mean(dim=2)  # (B, num_kv_heads, mid_len)
                # Blend mean + max over KV heads so single-head retrieval tokens remain protected
                attn_scores = 0.5 * per_head_mid.mean(dim=1) + 0.5 * per_head_mid.amax(dim=1)  # (B, mid_len)
                used_query_scoring = True
            except Exception:
                used_query_scoring = False

        if not used_query_scoring:
            # Fallback when called standalone without `register_query_hooks`
            probe_src = recent_k if recent_k.shape[-2] > 0 else mid_k
            probe = probe_src[:, :, -min(4, probe_src.shape[-2]) :, :].float()
            keys_float = mid_k.float()
            logits = torch.matmul(probe, keys_float.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_probs = torch.softmax(logits, dim=-1).mean(dim=-2)  # (B, H, M)
            v_norm = torch.norm(mid_v.float(), p=2, dim=-1)  # (B, H, M)
            v_norm = v_norm / (v_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6))
            step_scores = (attn_probs * v_norm).mean(dim=1)  # (B, M)
        else:
            v_norm = torch.norm(mid_v.float(), p=2, dim=-1).mean(dim=1)  # (B, M)
            v_norm = (v_norm / v_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(0.5, 2.0)
            step_scores = attn_scores * (0.75 + 0.25 * v_norm)

        # Combine with prior EMA scores (prior heavy-hitters occupy the start of mid_k)
        prev = self.importance_scores[layer_idx]
        if prev is not None and prev.shape[-1] > 0:
            if prev.shape[-1] == mid_len:
                step_scores = self.config.importance_decay * prev + step_scores
            elif prev.shape[-1] < mid_len:
                step_scores = step_scores.clone()
                step_scores[:, : prev.shape[-1]] += self.config.importance_decay * prev

        recency_bonus = torch.linspace(
            0.0,
            self.config.recency_weight,
            steps=mid_len,
            device=step_scores.device,
            dtype=step_scores.dtype,
        ).unsqueeze(0)
        return step_scores + recency_bonus

    @torch.no_grad()
    def update_from_attentions(self, attentions: Tuple[Optional[torch.Tensor], ...]) -> None:
        """Optional hook to incorporate explicit attention weights when output_attentions=True."""
        if not attentions:
            return
        for layer_idx, attn in enumerate(attentions):
            if attn is None or layer_idx >= len(self.key_cache):
                continue
            fp_len = self.key_cache[layer_idx].shape[-2]
            if fp_len == 0 or attn.shape[-1] < fp_len:
                continue
            attn_fp = attn[..., -fp_len:].float().mean(dim=(1, 2))  # (B, fp_len)
            self.importance_scores[layer_idx] = attn_fp

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the cumulative logical sequence length seen by `layer_idx`.
        Keeping this monotonic across KV compression preserves true RoPE `position_ids`
        in `Qwen2Model.forward` so compressed caches do not shift token positions.
        """
        idx = 0 if layer_idx is None else layer_idx
        if idx < len(self.total_tokens_seen_per_layer):
            return int(self.total_tokens_seen_per_layer[idx])
        return 0

    def get_active_kv_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the actual physical number of cached tokens (Hot FP + Warm INT4/INT8)."""
        idx = 0 if layer_idx is None else layer_idx
        fp_len = (
            int(self.key_cache[idx].shape[-2])
            if idx < len(self.key_cache) and self.key_cache[idx].numel() > 0
            else 0
        )
        q_block = self.quant_key_blocks[idx] if idx < len(self.quant_key_blocks) else None
        q_len = q_block.seq_length if q_block is not None else 0
        return fp_len + q_len

    def get_total_retained_tokens(self, layer_idx: int = 0) -> int:
        """Returns the sum of Hot-Tier (FP) tokens and Warm-Tier (INT4/INT8) tokens."""
        return self.get_active_kv_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: Optional[int] = 0) -> Tuple[int, int]:
        """
        Returns `(kv_length, kv_offset)` required by transformers >= 4.48 / 5.x mask builders.
        With `q_offset = seen` and `kv_offset = seen - actual_cached`, causal inequalities
        `kv_offset + kv_idx <= q_offset + q_idx` remain exact after token pruning.
        """
        actual_cached = self.get_active_kv_length(layer_idx)
        seen = self.get_seq_length(layer_idx)
        kv_length = actual_cached + query_length
        kv_offset = max(0, seen - actual_cached)
        return kv_length, kv_offset

    def get_query_offset(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the logical position offset of the next incoming query block."""
        return self.get_seq_length(layer_idx)

    def get_max_length(self, layer_idx: Optional[int] = None) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    @torch.no_grad()
    def crop(self, max_length: int) -> None:
        """
        Rolls back the KV cache to logical sequence length `max_length` across all layers in O(1)
        time. Used by Speculative Decoding to discard rejected tail draft tokens cleanly.
        """
        cur_seen = self.get_seq_length(0)
        if max_length < 0:
            max_length = cur_seen + max_length

        for layer_idx in range(len(self.key_cache)):
            layer_seen = self.total_tokens_seen_per_layer[layer_idx]
            if layer_seen <= max_length:
                continue

            tokens_to_remove = layer_seen - max_length
            fp_len = self._fp_len[layer_idx]
            keep_fp = max(0, fp_len - tokens_to_remove)
            self._fp_len[layer_idx] = keep_fp
            if self._k_buf[layer_idx] is not None:
                self.key_cache[layer_idx] = self._k_buf[layer_idx][:, :, :keep_fp, :]
                self.value_cache[layer_idx] = self._v_buf[layer_idx][:, :, :keep_fp, :]
            self.total_tokens_seen_per_layer[layer_idx] = max(0, layer_seen - tokens_to_remove)

        self._seen_tokens = self.get_seq_length(0)

    def should_compress(self) -> bool:
        """Returns True if the Hot FP Tier has reached the amortized compression threshold."""
        if not self.config.enabled or len(self.key_cache) == 0:
            return False
        seen_0 = self.get_seq_length(0)
        _, _, _, fp_budget, amortize_stride = self._effective_budgets(seen_0)
        return self.key_cache[0].shape[-2] >= fp_budget + amortize_stride

    @torch.no_grad()
    def compress_step(
        self,
        attentions: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        force: bool = False,
        shared_mid_scores: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Executes amortized adaptive KV-cache compression across all layers:
        1. Preserves the first `num_sink_tokens` (Attention Sinks) in full precision.
        2. Preserves the last `recent_window_size` tokens in full precision.
        3. Ranks middle tokens per layer by composite relevance score (attention alignment + value norm + recency),
           or uses `shared_mid_scores` when sharing a keep-set across target and draft (Step 5).
        4. Retains top Heavy-Hitters in the Hot FP Tier, immediately packs medium-importance tokens
           into the Warm INT4/INT8 Tier (`quant_key_blocks`/`quant_value_blocks`), and prunes
           low-importance / stale tokens. Preserves monotonic `_seen_tokens` for RoPE positions.
        """
        if not self.config.enabled or len(self.key_cache) == 0:
            return

        if attentions is not None:
            self.update_from_attentions(attentions)

        seen_0 = self.get_seq_length(0)
        _, hh_budget, quant_budget, fp_budget, amortize_stride = self._effective_budgets(seen_0)

        fp_len_0 = self.key_cache[0].shape[-2]
        if not force and fp_len_0 < fp_budget + amortize_stride:
            return
        if force and fp_len_0 <= fp_budget:
            return

        sink_size = self.config.num_sink_tokens
        recent_size = self.config.recent_window_size
        compressed_any = False

        for layer_idx in range(len(self.key_cache)):
            fp_k = self.key_cache[layer_idx]
            fp_v = self.value_cache[layer_idx]
            fp_len = fp_k.shape[-2]
            if fp_len <= fp_budget:
                continue

            actual_sink = min(sink_size, fp_len)
            actual_recent = min(recent_size, max(0, fp_len - actual_sink))
            mid_start = actual_sink
            mid_end = fp_len - actual_recent
            mid_len = mid_end - mid_start

            if mid_len <= hh_budget:
                continue

            mid_k = fp_k[:, :, mid_start:mid_end, :]
            mid_v = fp_v[:, :, mid_start:mid_end, :]
            recent_k = fp_k[:, :, mid_end:, :] if actual_recent > 0 else fp_k[:, :, :0, :]

            if shared_mid_scores is not None and int(shared_mid_scores.shape[-1]) == mid_len:
                composite_scores = (
                    shared_mid_scores.reshape(1, mid_len)
                    .expand(fp_k.shape[0], -1)
                    .to(device=fp_k.device, dtype=torch.float32)
                )
            else:
                composite_scores = self._compute_token_relevance(
                    layer_idx,
                    mid_k,
                    mid_v,
                    recent_k,
                    fp_k=fp_k,
                    actual_sink=actual_sink,
                    mid_end=mid_end,
                )
            self._obs_inputs[layer_idx] = None

            keep_hh = min(hh_budget, mid_len)
            remaining_mid = mid_len - keep_hh
            if not compressed_any:
                compressed_any = True
                self.compress_events += 1
                self._last_layer_mid_scores = [None] * len(self._last_layer_mid_scores)
                self._last_compress_geom = (int(fp_len), int(actual_sink), int(mid_end), int(keep_hh))
            promote_to_quant = min(quant_budget, remaining_mid)

            # Use batch-mean relevance so B > 1 autoregressive batching selects shared indices cleanly.
            batch_scores = composite_scores.mean(dim=0) if composite_scores.shape[0] > 1 else composite_scores[0]
            self._last_layer_mid_scores[layer_idx] = batch_scores.detach()
            sorted_indices = torch.argsort(batch_scores, dim=-1, descending=True)
            hh_idx_1d, _ = torch.sort(sorted_indices[:keep_hh], dim=-1)
            self.importance_scores[layer_idx] = composite_scores[:, hh_idx_1d]

            dev = fp_k.device
            sink_idx_1d = torch.arange(0, actual_sink, device=dev, dtype=torch.long)
            recent_idx_1d = torch.arange(mid_end, fp_len, device=dev, dtype=torch.long)
            keep_fp_idx_1d = torch.cat([sink_idx_1d, hh_idx_1d + mid_start, recent_idx_1d], dim=0)
            self._last_keep_indices[layer_idx] = keep_fp_idx_1d.detach()

            dropped_from_quant = 0
            if promote_to_quant > 0 and quant_budget > 0:
                quant_idx_1d, _ = torch.sort(
                    sorted_indices[keep_hh : keep_hh + promote_to_quant], dim=-1
                )
                quant_idx_1d = quant_idx_1d + mid_start
                new_q_k_block = quantize_tensor(
                    fp_k.index_select(2, quant_idx_1d),
                    bits=self.config.key_quant_bits,
                    axis="per_channel",
                )
                new_q_v_block = quantize_tensor(
                    fp_v.index_select(2, quant_idx_1d),
                    bits=self.config.quant_bits,
                    axis="per_token",
                )

                existing_q_k = self.quant_key_blocks[layer_idx]
                existing_q_v = self.quant_value_blocks[layer_idx]
                old_q_len = existing_q_k.seq_length if existing_q_k is not None else 0

                if existing_q_k is not None and existing_q_k.seq_length > 0:
                    self.quant_key_blocks[layer_idx] = concat_quantized_blocks(
                        existing_q_k, new_q_k_block, quant_budget
                    )
                    self.quant_value_blocks[layer_idx] = concat_quantized_blocks(
                        existing_q_v, new_q_v_block, quant_budget
                    )
                else:
                    if new_q_k_block.seq_length > quant_budget:
                        new_q_k_block = concat_quantized_blocks(
                            quantize_tensor(
                                fp_k[:, :, :0, :], bits=self.config.key_quant_bits, axis="per_channel"
                            ),
                            new_q_k_block,
                            quant_budget,
                        )
                        new_q_v_block = concat_quantized_blocks(
                            quantize_tensor(
                                fp_v[:, :, :0, :], bits=self.config.quant_bits, axis="per_token"
                            ),
                            new_q_v_block,
                            quant_budget,
                        )
                    self.quant_key_blocks[layer_idx] = new_q_k_block
                    self.quant_value_blocks[layer_idx] = new_q_v_block

                new_q_len = self.quant_key_blocks[layer_idx].seq_length
                dropped_from_quant = max(0, old_q_len + promote_to_quant - new_q_len)
                self._warm_k_dequant[layer_idx] = None
                self._warm_v_dequant[layer_idx] = None

            evicted_from_mid = remaining_mid - promote_to_quant
            self.pruned_tokens_per_layer[layer_idx] += evicted_from_mid + dropped_from_quant

            # Single C++ index_select call to form Hot Tier: [Sinks | Heavy-Hitters | Recent Window]
            new_fp_k = fp_k.index_select(2, keep_fp_idx_1d).contiguous()
            new_fp_v = fp_v.index_select(2, keep_fp_idx_1d).contiguous()
            kept_len = new_fp_k.shape[-2]
            buf_k = self._k_buf[layer_idx]
            if buf_k is not None and buf_k.shape[-2] <= kept_len + 512:
                buf_k[:, :, :kept_len, :].copy_(new_fp_k)
                self._v_buf[layer_idx][:, :, :kept_len, :].copy_(new_fp_v)
            else:
                b, h, _, d = new_fp_k.shape
                new_cap = kept_len + 128
                nbk = torch.empty((b, h, new_cap, d), dtype=new_fp_k.dtype, device=new_fp_k.device)
                nbv = torch.empty((b, h, new_cap, d), dtype=new_fp_v.dtype, device=new_fp_v.device)
                nbk[:, :, :kept_len, :].copy_(new_fp_k)
                nbv[:, :, :kept_len, :].copy_(new_fp_v)
                self._k_buf[layer_idx] = nbk
                self._v_buf[layer_idx] = nbv
            self._fp_len[layer_idx] = kept_len
            self.key_cache[layer_idx] = self._k_buf[layer_idx][:, :, :kept_len, :]
            self.value_cache[layer_idx] = self._v_buf[layer_idx][:, :, :kept_len, :]

    def get_shared_mid_scores(self) -> Optional[torch.Tensor]:
        """
        Step 5: Returns a layer-consensus normalized middle-token relevance vector (1D)
        from the most recent `compress_step()` so a draft cache can share the target's keep-set.
        """
        valid = [s.float() for s in self._last_layer_mid_scores if s is not None and s.numel() > 0]
        if not valid:
            return None
        target_len = valid[0].shape[-1]
        same_len = [s for s in valid if s.shape[-1] == target_len]
        if not same_len:
            return None
        normed = []
        for s in same_len:
            s_min = s.min()
            s_max = s.max()
            span = (s_max - s_min).clamp(min=1e-6)
            normed.append((s - s_min) / span)
        return torch.stack(normed, dim=0).mean(dim=0)

    def get_shared_keep_indices(self) -> Optional[torch.Tensor]:
        """
        Draft keep-set for `--draft-cache shared_keepset`, valid right after a compress_step() that evicted.
        Target layers each keep their own heavy hitters, so "the same positions" is defined as the target's
        layer-consensus keep-set: the same sinks and recent window as every target layer, plus the top-`keep_hh`
        middle tokens by the mean (min-max normalised) target relevance over all target layers. Same count as
        each target layer. Returned as sorted physical indices into the pre-compression buffer of length fp_len.
        """
        if self._last_compress_geom is None:
            return None
        fp_len, actual_sink, mid_end, keep_hh = self._last_compress_geom
        scores = self.get_shared_mid_scores()
        mid_len = mid_end - actual_sink
        if scores is None or int(scores.shape[-1]) != mid_len:
            return None
        dev = scores.device
        hh = torch.sort(torch.argsort(scores, descending=True)[:keep_hh])[0] + actual_sink
        return torch.cat(
            [
                torch.arange(0, actual_sink, device=dev, dtype=torch.long),
                hh.to(torch.long),
                torch.arange(mid_end, fp_len, device=dev, dtype=torch.long),
            ]
        )

    def get_last_keep_indices(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """Returns the last selected Hot-tier keep indices for `layer_idx` (if compressed)."""
        if 0 <= layer_idx < len(self._last_keep_indices):
            return self._last_keep_indices[layer_idx]
        return None

    @torch.no_grad()
    def finalize_quantization(self) -> None:
        """Retained for API compatibility; Warm Tier tokens are now packed eagerly during `compress_step`."""
        return

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Computes exact memory consumption (in bytes and MB) of the Adaptive KV-Cache
        (Hot FP Tier + Warm Quantized Tier) compared against an uncompressed full-precision
        KV cache of all seen tokens.
        """
        total_actual_bytes = 0
        total_uncompressed_bytes = 0
        num_layers = len(self.key_cache)

        if num_layers == 0:
            return {
                "num_layers": 0,
                "active_seq_len": 0,
                "fp_tokens": 0,
                "quantized_tokens": 0,
                "pruned_tokens": 0,
                "total_tokens_seen": 0,
                "actual_memory_mb": 0.0,
                "uncompressed_memory_mb": 0.0,
                "compression_ratio": 1.0,
                "token_compression_ratio": 1.0,
                "token_retention_pct": 100.0,
                "memory_saved_pct": 0.0,
            }

        for layer_idx in range(num_layers):
            fp_k = self.key_cache[layer_idx]
            fp_v = self.value_cache[layer_idx]
            fp_bytes = (
                fp_k.numel() * fp_k.element_size() + fp_v.numel() * fp_v.element_size()
            )

            q_k = self.quant_key_blocks[layer_idx]
            q_v = self.quant_value_blocks[layer_idx]
            q_bytes = (q_k.memory_bytes() if q_k else 0) + (q_v.memory_bytes() if q_v else 0)

            total_actual_bytes += fp_bytes + q_bytes

            batch, heads, _, head_dim = fp_k.shape
            elem_size = fp_k.element_size()
            seen = self.total_tokens_seen_per_layer[layer_idx]
            uncompressed_layer_bytes = 2 * batch * heads * seen * head_dim * elem_size
            total_uncompressed_bytes += uncompressed_layer_bytes

        fp_tokens_0 = self.key_cache[0].shape[-2]
        q_tokens_0 = self.quant_key_blocks[0].seq_length if self.quant_key_blocks[0] else 0
        seen_0 = self.total_tokens_seen_per_layer[0]
        pruned_0 = max(0, seen_0 - (fp_tokens_0 + q_tokens_0))

        actual_mb = total_actual_bytes / (1024 * 1024)
        uncompressed_mb = total_uncompressed_bytes / (1024 * 1024)
        compression_ratio = (
            total_uncompressed_bytes / max(1, total_actual_bytes)
            if total_uncompressed_bytes > 0
            else 1.0
        )
        saved_pct = (
            (1.0 - (total_actual_bytes / total_uncompressed_bytes)) * 100.0
            if total_uncompressed_bytes > 0
            else 0.0
        )

        active_0 = fp_tokens_0 + q_tokens_0
        token_comp_ratio = (seen_0 / max(1, active_0)) if seen_0 > 0 else 1.0
        token_ret_pct = (active_0 / max(1, seen_0)) * 100.0 if seen_0 > 0 else 100.0

        return {
            "num_layers": num_layers,
            "active_seq_len": active_0,
            "fp_tokens": fp_tokens_0,
            "quantized_tokens": q_tokens_0,
            "pruned_tokens": pruned_0,
            "total_tokens_seen": seen_0,
            "actual_memory_mb": round(actual_mb, 4),
            "uncompressed_memory_mb": round(uncompressed_mb, 4),
            "compression_ratio": round(compression_ratio, 3),
            "token_compression_ratio": round(token_comp_ratio, 3),
            "token_retention_pct": round(token_ret_pct, 2),
            "memory_saved_pct": round(saved_pct, 2),
        }

"""
Draft-model attention backend: split-KV ("flash-decoding" style) attention for short-query decode.

Why the draft does not use FlashAttention-2:
  The draft decodes against a pre-allocated `StaticCache`. With `flash_attention_2`, HF must pass a 2D
  padding mask so FA2 only reads the valid prefix of the static buffer. Every layer then runs
  `_upad_input` -> `_get_unpad_data`, which calls `torch.nonzero(...)` and `.max().item()` (two host syncs
  per layer, 48 per draft token for Qwen2.5-0.5B) and gathers the whole cached K/V. Those ops are
  data-dependent, so they are graph breaks and the step cannot be captured in a CUDA graph.

Why the draft no longer calls SDPA for decode (q_len <= 8):
  On a StaticCache, HF always builds an explicit (B, 1, q_len, max_cache_len) boolean mask for q_len == 1
  (`create_causal_mask` disables the is_causal skip for compileable caches). With a mask, SDPA cannot use
  the flash backend and picks the memory-efficient kernel. That kernel parallelises over
  (batch x heads x query-blocks) only. For the 0.5B draft that is 1 x 2 KV heads x 1 block = 2 thread blocks
  per layer, each streaming the whole cache serially on one SM. Cost grows linearly with cache capacity
  while the rest of the GPU idles. That is the context-dependent draft overhead (7.9 ms at 1K -> 27 ms at 8K)
  even though the draft KV is only ~100 MB.

What this backend does instead (plain torch ops, CUDA-graph / torch.compile friendly, no custom kernel):
  For q_len <= 8 the query is packed per KV head (B, Hkv, n_rep * q_len, D) (no repeat_kv copy), and the
  KV axis is split into S chunks: K/V (B, Hkv, L, D) -> (B, Hkv, S, L/S, D) (a free view). Scores and P@V
  are batched matmuls over B*Hkv*S independent problems, so every SM gets work, and the S partial outputs
  are summed after a global softmax (exact up to float rounding). Prefill / longer queries still delegate
  to HF's standard SDPA implementation unchanged.
"""

from typing import Any, Dict, Optional

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

DRAFT_ATTN_IMPL = "sdpa_gqa_decode"

# Queries up to this length (draft decode q_len=1, catch-up q_len=2, short verify buckets) use split-KV.
SPLIT_KV_MAX_Q = 8
# Target keys per split. Static capacities are multiples of 8, so some chunk size in the ladder always divides.
_SPLIT_CHUNK_LADDER = (256, 128, 64, 32, 16, 8)

# Last eager (non-traced) decode call, for diagnostics only. Never written while torch.compile traces.
LAST_DECODE_INFO: Dict[str, Any] = {}


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # pragma: no cover - very old torch
        return False


def _pick_chunk(kv_len: int) -> int:
    for c in _SPLIT_CHUNK_LADDER:
        if kv_len >= 2 * c and kv_len % c == 0:
            return c
    return kv_len  # one split (short or odd-length dynamic caches)


def split_kv_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    n_rep: int,
    scaling: Optional[float],
) -> torch.Tensor:
    """
    query (B, Hq, q, D), key/value (B, Hkv, L, D), attention_mask None | bool (True = attend) | additive float,
    broadcastable to (B, 1, q, L). Returns (B, q, Hq, Dv) like HF attention backends.
    """
    b, hq, q_len, d = query.shape
    hkv, kv_len, dv = key.shape[1], key.shape[2], value.shape[-1]
    scale = float(scaling) if scaling is not None else d ** -0.5

    # Head h uses KV head h // n_rep (repeat_kv order) -> rows ordered (rep, q) inside each KV head.
    qp = query.reshape(b, hkv, n_rep * q_len, d)
    rows = n_rep * q_len

    chunk = _pick_chunk(kv_len)
    n_split = kv_len // chunk
    k_s = key.reshape(b, hkv, n_split, chunk, d)
    v_s = value.reshape(b, hkv, n_split, chunk, dv)

    # (B, Hkv, S, rows, chunk): B*Hkv*S independent small GEMMs
    scores = torch.matmul(qp.unsqueeze(2), k_s.transpose(-1, -2)).float() * scale

    if attention_mask is not None:
        m = attention_mask[..., :kv_len]
        # (B|1, 1, q, L) -> (B|1, 1, 1, rep*q, L) -> split view (B|1, 1, S, rep*q, chunk)
        m = m.reshape(m.shape[0], 1, 1, m.shape[-2], kv_len).expand(m.shape[0], 1, n_rep, m.shape[-2], kv_len)
        if m.shape[-2] != q_len:  # mask broadcast over queries
            m = m.expand(m.shape[0], 1, n_rep, q_len, kv_len)
        m = m.reshape(m.shape[0], 1, rows, n_split, chunk).transpose(2, 3)
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()

    # Global (exact) softmax over all splits, then per-split P@V and a sum over splits.
    row_max = scores.amax(dim=(2, 4), keepdim=True)
    p = torch.exp(scores - row_max)
    denom = p.sum(dim=(2, 4), keepdim=True)  # (B, Hkv, 1, rows, 1)
    out = torch.matmul(p.to(value.dtype), v_s).float()  # (B, Hkv, S, rows, Dv)
    out = (out.sum(dim=2, keepdim=True) / denom).squeeze(2)  # (B, Hkv, rows, Dv)
    out = out.to(value.dtype).reshape(b, hkv, n_rep, q_len, dv).reshape(b, hq, q_len, dv)
    return out.transpose(1, 2).contiguous()


def sdpa_gqa_decode_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    dropout: float = 0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    n_rep = int(getattr(module, "num_key_value_groups", 1) or 1)
    q_len = int(query.shape[2])
    if (
        q_len <= SPLIT_KV_MAX_Q
        and kwargs.get("position_bias") is None
        and (attention_mask is None or (attention_mask.dim() == 4 and attention_mask.shape[1] == 1))
        and query.shape[1] == key.shape[1] * n_rep
    ):
        if attention_mask is None and q_len > 1:
            # No mask + q_len > 1 means plain causal over the trailing q_len keys (bottom-right aligned).
            kv_len = int(key.shape[2])
            q_idx = torch.arange(q_len, device=query.device).view(1, 1, q_len, 1) + (kv_len - q_len)
            attention_mask = torch.arange(kv_len, device=query.device).view(1, 1, 1, kv_len) <= q_idx
        if not _is_compiling():
            LAST_DECODE_INFO.update(
                path="split_kv_matmul",
                q_len=q_len,
                kv_len=int(key.shape[2]),
                mask_shape=tuple(attention_mask.shape) if attention_mask is not None else None,
                n_split=int(key.shape[2]) // _pick_chunk(int(key.shape[2])),
            )
        return split_kv_decode_attention(query, key, value, attention_mask, n_rep, scaling), None
    if not _is_compiling():
        LAST_DECODE_INFO.update(
            path="hf_sdpa", q_len=q_len, kv_len=int(key.shape[2]),
            mask_shape=tuple(attention_mask.shape) if attention_mask is not None else None, n_split=None,
        )
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )


AttentionInterface.register(DRAFT_ATTN_IMPL, sdpa_gqa_decode_forward)
AttentionMaskInterface.register(DRAFT_ATTN_IMPL, sdpa_mask)


def use_packed_gqa_draft_attention(model) -> str:
    """Switches an SDPA-loaded draft model to the split-KV decode variant. Returns the active backend name."""
    if model is not None and getattr(model.config, "_attn_implementation", None) == "sdpa":
        model.config._attn_implementation = DRAFT_ATTN_IMPL
    return str(getattr(model.config, "_attn_implementation", None)) if model is not None else "none"

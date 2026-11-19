"""
Per-token asymmetric INT8 and packed INT4 KV-Cache Quantizer.

Compresses low/medium-importance Key and Value states along the head_dim axis,
storing packed uint8/int8 representations along with per-token scale and zero-point
metadata to genuinely reduce GPU memory consumption.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class QuantizedKVBlock:
    """Compact quantized representation of a Key or Value tensor slice."""

    q_data: torch.Tensor       # uint8 (packed for 4-bit, or raw for 8-bit)
    scale: torch.Tensor        # orig_dtype scale: (B, H, S, 1) per_token or (B, H, 1, D) per_channel
    zero_point: torch.Tensor   # orig_dtype min-val: matching scale shape
    orig_dtype: torch.dtype
    orig_head_dim: int
    bits: int
    axis: str = "per_token"
    segments: Optional[Tuple["QuantizedKVBlock", ...]] = None

    @property
    def seq_length(self) -> int:
        if self.segments is not None:
            return sum(seg.seq_length for seg in self.segments)
        return int(self.q_data.shape[2])

    def memory_bytes(self) -> int:
        """Exact physical memory footprint of the quantized block in bytes."""
        if self.segments is not None:
            return sum(seg.memory_bytes() for seg in self.segments)
        return (
            self.q_data.numel() * self.q_data.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero_point.numel() * self.zero_point.element_size()
        )

    def dequantize(self) -> torch.Tensor:
        """Reconstructs the approximate full-precision tensor for attention computation."""
        return dequantize_tensor(self)


def quantize_tensor(
    x: torch.Tensor,
    bits: int = 4,
    axis: str = "per_token",
) -> QuantizedKVBlock:
    """
    Quantizes a 4D KV tensor of shape (batch, num_heads, seq_len, head_dim) using
    asymmetric uniform quantization.

    - `axis="per_token"` (along `head_dim`): ideal for Value states.
    - `axis="per_channel"` (along `seq_len`): neutralizes fixed channel-wise outliers
      in Qwen2.5/Llama Key states (KIVI-style).
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bit-width: {bits}. Choose 4 or 8.")
    if axis not in ("per_token", "per_channel"):
        raise ValueError(f"Unsupported quantization axis: {axis}. Choose 'per_token' or 'per_channel'.")

    orig_dtype = x.dtype
    batch, heads, seq_len, head_dim = x.shape
    meta_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.float32

    if seq_len == 0:
        empty_q = torch.empty(
            (batch, heads, 0, head_dim // 2 if bits == 4 else head_dim),
            dtype=torch.uint8,
            device=x.device,
        )
        meta_shape = (batch, heads, 0, 1) if axis == "per_token" else (batch, heads, 1, head_dim)
        empty_meta = torch.zeros(meta_shape, dtype=meta_dtype, device=x.device)
        return QuantizedKVBlock(empty_q, empty_meta, empty_meta, orig_dtype, head_dim, bits, axis)

    q_max = float((1 << bits) - 1)  # 15.0 for 4-bit, 255.0 for 8-bit
    x_float = x.float()
    reduce_dim = -1 if axis == "per_token" else 2

    x_min = x_float.amin(dim=reduce_dim, keepdim=True)
    x_max = x_float.amax(dim=reduce_dim, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-6) / q_max

    q_int = torch.round((x_float - x_min) / scale).clamp(0.0, q_max).to(torch.uint8)

    if bits == 4:
        if head_dim % 2 != 0:
            pad = torch.zeros((batch, heads, seq_len, 1), dtype=torch.uint8, device=x.device)
            q_int = torch.cat([q_int, pad], dim=-1)
        high_nibble = q_int[..., 0::2] << 4
        low_nibble = q_int[..., 1::2] & 0x0F
        q_packed = (high_nibble | low_nibble).contiguous()
    else:
        q_packed = q_int.contiguous()

    return QuantizedKVBlock(
        q_data=q_packed,
        scale=scale.to(meta_dtype),
        zero_point=x_min.to(meta_dtype),
        orig_dtype=orig_dtype,
        orig_head_dim=head_dim,
        bits=bits,
        axis=axis,
    )


def dequantize_tensor(block: QuantizedKVBlock) -> torch.Tensor:
    """
    Dequantize a QuantizedKVBlock back to its original shape and dtype directly in `orig_dtype`
    (avoiding full-tensor float32 temporary allocations on GPU).
    """
    if block.segments is not None:
        if len(block.segments) == 0 or block.seq_length == 0:
            batch, heads, _, _ = block.q_data.shape
            return torch.empty(
                (batch, heads, 0, block.orig_head_dim),
                dtype=block.orig_dtype,
                device=block.q_data.device,
            )
        if len(block.segments) == 1:
            return dequantize_tensor(block.segments[0])
        return torch.cat([dequantize_tensor(seg) for seg in block.segments], dim=2)

    if block.seq_length == 0:
        batch, heads, _, _ = block.q_data.shape
        return torch.empty(
            (batch, heads, 0, block.orig_head_dim),
            dtype=block.orig_dtype,
            device=block.q_data.device,
        )

    out_dtype = block.orig_dtype
    if block.bits == 4:
        high = (block.q_data >> 4) & 0x0F
        low = block.q_data & 0x0F
        unpacked = torch.stack([high, low], dim=-1).flatten(start_dim=-2)
        if unpacked.shape[-1] > block.orig_head_dim:
            unpacked = unpacked[..., : block.orig_head_dim]
        q_vals = unpacked.to(out_dtype)
    else:
        q_vals = block.q_data.to(out_dtype)

    scale = block.scale.to(out_dtype) if block.scale.dtype != out_dtype else block.scale
    zp = block.zero_point.to(out_dtype) if block.zero_point.dtype != out_dtype else block.zero_point
    return q_vals * scale + zp


def _slice_leaf_tail(block: QuantizedKVBlock, keep_len: int) -> QuantizedKVBlock:
    """Slices the trailing `keep_len` tokens from a single quantized block without re-quantizing."""
    if block.seq_length <= keep_len:
        return block
    q_data = block.q_data[:, :, -keep_len:, :].contiguous()
    if block.axis == "per_token":
        scale = block.scale[:, :, -keep_len:, :].contiguous()
        zero_point = block.zero_point[:, :, -keep_len:, :].contiguous()
    else:
        # per_channel scale/zero_point apply uniformly along dim=2
        scale = block.scale
        zero_point = block.zero_point
    return QuantizedKVBlock(
        q_data=q_data,
        scale=scale,
        zero_point=zero_point,
        orig_dtype=block.orig_dtype,
        orig_head_dim=block.orig_head_dim,
        bits=block.bits,
        axis=block.axis,
    )


def concat_quantized_blocks(
    existing: QuantizedKVBlock,
    new_block: QuantizedKVBlock,
    max_seq_len: int,
) -> QuantizedKVBlock:
    """
    Concatenates two QuantizedKVBlocks along `dim=2` and trims to `max_seq_len`.
    For `per_token`, concatenates directly without dequantization; for `per_channel`,
    preserves chunk segments so previously quantized chunks never suffer repeated re-quantization drift.
    """
    if existing.seq_length == 0:
        if new_block.seq_length <= max_seq_len:
            return new_block
        if new_block.segments is not None:
            return concat_quantized_blocks(new_block, QuantizedKVBlock(
                q_data=new_block.q_data[:, :, :0, :],
                scale=new_block.scale,
                zero_point=new_block.zero_point,
                orig_dtype=new_block.orig_dtype,
                orig_head_dim=new_block.orig_head_dim,
                bits=new_block.bits,
                axis=new_block.axis,
            ), max_seq_len)
        return _slice_leaf_tail(new_block, max_seq_len)

    if existing.axis == "per_channel" or new_block.axis == "per_channel":
        raw_segs: List[QuantizedKVBlock] = []
        for blk in (existing, new_block):
            if blk.segments is not None:
                raw_segs.extend(s for s in blk.segments if s.seq_length > 0)
            elif blk.seq_length > 0:
                raw_segs.append(blk)

        # Keep only the trailing `max_seq_len` tokens across segments (never re-quantizing)
        kept_rev: List[QuantizedKVBlock] = []
        remaining = max_seq_len
        for seg in reversed(raw_segs):
            if remaining <= 0:
                break
            if seg.seq_length <= remaining:
                kept_rev.append(seg)
                remaining -= seg.seq_length
            else:
                kept_rev.append(_slice_leaf_tail(seg, remaining))
                remaining = 0
        kept = list(reversed(kept_rev))
        if len(kept) == 1:
            return kept[0]
        head_ref = kept[-1]
        return QuantizedKVBlock(
            q_data=head_ref.q_data[:, :, :0, :],
            scale=head_ref.scale[:, :, :0, :],
            zero_point=head_ref.zero_point[:, :, :0, :],
            orig_dtype=head_ref.orig_dtype,
            orig_head_dim=head_ref.orig_head_dim,
            bits=head_ref.bits,
            axis=head_ref.axis,
            segments=tuple(kept),
        )

    q_data = torch.cat([existing.q_data, new_block.q_data], dim=2)
    scale = torch.cat([existing.scale, new_block.scale], dim=2)
    zero_point = torch.cat([existing.zero_point, new_block.zero_point], dim=2)

    if q_data.shape[2] > max_seq_len:
        q_data = q_data[:, :, -max_seq_len:, :].contiguous()
        scale = scale[:, :, -max_seq_len:, :].contiguous()
        zero_point = zero_point[:, :, -max_seq_len:, :].contiguous()

    return QuantizedKVBlock(
        q_data=q_data,
        scale=scale,
        zero_point=zero_point,
        orig_dtype=new_block.orig_dtype,
        orig_head_dim=new_block.orig_head_dim,
        bits=new_block.bits,
        axis=new_block.axis,
    )

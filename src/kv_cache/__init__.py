"""Adaptive KV-Cache Compression module (Pruning + Mixed-Precision Quantization)."""

from src.kv_cache.adaptive_cache import AdaptiveKVCache, LayerCacheStats
from src.kv_cache.quantizer import QuantizedKVBlock, dequantize_tensor, quantize_tensor

__all__ = [
    "AdaptiveKVCache",
    "LayerCacheStats",
    "QuantizedKVBlock",
    "quantize_tensor",
    "dequantize_tensor",
]

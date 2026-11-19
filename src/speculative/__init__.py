"""Speculative Decoding and Baseline Autoregressive Decoding pipelines."""

from src.speculative.baseline_decoder import GenerationOutput, autoregressive_generate
from src.speculative.speculative_decoder import speculative_generate

__all__ = [
    "GenerationOutput",
    "autoregressive_generate",
    "speculative_generate",
]

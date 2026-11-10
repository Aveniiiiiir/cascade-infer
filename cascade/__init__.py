"""Cascade: production-grade LLM inference engine.

PyTorch + Triton implementation of the DeepInfer design: continuous batching,
paged KV cache with prefix caching, zero-overhead scheduling (ZOS), MTP
speculative decoding with confidence-scheduled variable-length verification,
batch-invariant deterministic sampling, and structured outputs.
"""

from cascade.common import envs

__version__ = "0.1.0"

__all__ = ["__version__", "envs"]

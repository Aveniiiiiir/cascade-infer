"""Centralized environment-variable registry.

Mirrors DeepInfer's `deepinfer/environ.py` pattern: descriptor-based, lazily
parsed on first access, grouped in a single `envs` namespace so that the full
knob surface of the engine is discoverable in one place.
"""

from __future__ import annotations

import os
from typing import Callable


class _EnvDescriptor:
    """Lazily-parsed environment variable bound to a class attribute."""

    def __init__(self, key: str, parser: Callable[[str], object], default: object) -> None:
        self.key = key
        self.parser = parser
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr = name

    def __get__(self, obj: object, objtype: type | None = None) -> object:
        raw = os.environ.get(self.key)
        if raw is None or raw == "":
            return self.default
        try:
            return self.parser(raw)
        except ValueError as e:
            raise ValueError(f"invalid value for {self.key}: {raw!r} ({e})") from e


class _EnvNamespace:
    """All Cascade environment knobs. Values are read lazily on each access."""

    # --- cache -----------------------------------------------------------------
    #: tokens per KV block. 64 matches FlashMLA page geometry and Triton tile
    #: shapes comfortably; DeepInfer uses 128 (KVSTORE_GRAIN) for MXFP8 packing.
    BLOCK_SIZE = _EnvDescriptor("CASCADE_BLOCK_SIZE", int, 64)

    # --- scheduler -------------------------------------------------------------
    #: maximum number of sequences in a decode batch.
    MAX_NUM_SEQS = _EnvDescriptor("CASCADE_MAX_NUM_SEQS", int, 256)
    #: maximum number of tokens (prefill chunk + decode) scheduled per step.
    MAX_NUM_BATCHED_TOKENS = _EnvDescriptor("CASCADE_MAX_NUM_BATCHED_TOKENS", int, 8192)
    #: enable prefix caching (block hash chain + LRU).
    PREFIX_CACHING = _EnvDescriptor("CASCADE_PREFIX_CACHING", int, 1)
    #: recompute-style preemption (DeepInfer semantics); 0 = stop preempted
    #: requests instead of resetting them for recompute.
    PREEMPTION_RECOMPUTE = _EnvDescriptor("CASCADE_PREEMPTION_RECOMPUTE", int, 1)

    # --- determinism -----------------------------------------------------------
    #: batch-invariant inference switch (four要素: frozen kernel configs, no
    #: non-invariant fusions, single-reduction EP, varlen-MTP disabled).
    BATCH_INVARIANT = _EnvDescriptor("CASCADE_BATCH_INVARIANT", int, 0)

    # --- MTP / speculative decoding --------------------------------------------
    #: maximum speculative draft length K (DeepInfer: DEEPINFER_USE_NEXTN_DECODE, default 5).
    NEXTN_DECODE = _EnvDescriptor("CASCADE_NEXTN_DECODE", int, 3)
    #: confidence-scheduled variable-length verification
    #: (DeepInfer: DEEPINFER_ENABLE_MTP_VARLEN). Mutually exclusive with
    #: BATCH_INVARIANT, as in DeepInfer.
    ENABLE_MTP_VARLEN = _EnvDescriptor("CASCADE_ENABLE_MTP_VARLEN", int, 0)

    # --- ZOS --------------------------------------------------------------------
    #: zero-overhead scheduling: overlap CPU batch scheduling with device
    #: execution via the three-segment step pipeline.
    ZERO_OVERHEAD_SCHEDULING = _EnvDescriptor("CASCADE_ZERO_OVERHEAD_SCHEDULING", int, 1)

    # --- CUDA Graph ---------------------------------------------------------------
    #: decode forward 图捕获（无 triton/fp32 环境自动关闭）。
    CUDAGRAPH = _EnvDescriptor("CASCADE_CUDAGRAPH", int, 1)
    #: 档位间隔（2 的幂到 8 后按此等差；DeepInfer interval 默认 8）。
    CUDAGRAPH_BATCH_SIZE_INTERVAL = _EnvDescriptor("CASCADE_CUDAGRAPH_BATCH_SIZE_INTERVAL", int, 8)
    #: 图内 decode 注意力的固定 split 数（避免 seqlens.max().item() 同步）。
    CUDAGRAPH_NUM_SPLITS = _EnvDescriptor("CASCADE_CUDAGRAPH_NUM_SPLITS", int, 8)

    # --- misc -------------------------------------------------------------------
    #: 强制 Triton 算子全部走 PyTorch 参考路径（新架构 Triton 编译异常时的
    #: 逃生门；慢，但保证引擎可运行。图捕获需同时关 CASCADE_CUDAGRAPH）。
    DISABLE_TRITON = _EnvDescriptor("CASCADE_DISABLE_TRITON", int, 0)
    #: 1 = MLA 吸收路径（生产默认）；0 = 显式路径（与 HF 数值路径一致，
    #: 用于逐 token e2e 对拍；更慢）
    ABSORBED = _EnvDescriptor("CASCADE_ABSORBED", int, 1)
    #: regenerate golden files instead of comparing against them (tests only).
    REGEN_GOLDEN = _EnvDescriptor("CASCADE_REGEN_GOLDEN", int, 0)
    #: ZOS invariant checker in the hot loop (debug only).
    DEBUG_ZOS = _EnvDescriptor("CASCADE_DEBUG_ZOS", int, 0)


envs = _EnvNamespace()

__all__ = ["envs"]

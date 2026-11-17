"""KV cache engine: GPU-side paged KV tensors.

The layout math lives in `cascade.worker.kv_layout` (torch-free). This module
owns the actual device tensors and the reference scatter/gather paths used
until (and alongside) the Triton kernels land. Block ids are shared across
layers — a block table row addresses `[layer, block, slot, :]`.
"""

from __future__ import annotations

import torch

from cascade.common.logger import get_logger
from cascade.config import CacheConfig, ModelConfig
from cascade.worker.kv_layout import kv_shape

_log = get_logger("worker.cache_engine")


class CacheEngine:
    __slots__ = ("model_cfg", "cache_cfg", "num_blocks", "dtype", "device", "kv_cache")

    def __init__(
        self,
        model_cfg: ModelConfig,
        cache_cfg: CacheConfig,
        num_blocks: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_cfg = model_cfg
        self.cache_cfg = cache_cfg
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device
        shape = kv_shape(model_cfg, num_blocks, cache_cfg.block_size)
        # 单块扁平分配（DeepInfer cuMemAlloc 直分配的简化版；无 cuMem* 依赖）
        self.kv_cache = torch.zeros(shape, dtype=dtype, device=device)
        _log.info(
            "KV cache allocated: %s blocks × %d tokens/block, shape %s, %.2f GiB",
            num_blocks,
            cache_cfg.block_size,
            tuple(shape),
            self.kv_cache.numel() * self.kv_cache.element_size() / 2**30,
        )

    def layer_view(self, layer_idx: int) -> torch.Tensor:
        """第 layer_idx 层的视图: [num_blocks + 1, block_size, kv_cache_dim]。

        行 0 是保留的空块（BlockManager 的 block id 为 1..num_blocks）。
        """
        return self.kv_cache[layer_idx]

    # ------------------------------------------------------------------ scatter/gather
    # 写入路径经 triton_ops 分发（CUDA+triton → kernel；否则 PyTorch ref）。
    # 读取参考路径保留在本文件（分页注意力 kernel 落地后由 kernel 直读替换）。

    def store_latent(
        self,
        layer_idx: int,
        latent: torch.Tensor,  # [T, kv_cache_dim] 本步新 token 的 c‖rope
        seq_ids: torch.Tensor,  # [T] int64: 每 token 所属行（batch 内 seq 序号）
        positions: torch.Tensor,  # [T] int64: 绝对位置
        block_tables: torch.Tensor,  # [n_seqs, max_blocks] int64
    ) -> None:
        from cascade.model_executor.triton_ops.store_paged_kv import store_paged_kv

        kv = self.layer_view(layer_idx)
        store_paged_kv(kv, latent.to(kv.dtype), seq_ids, positions, block_tables)

    def gather_positions_reference(
        self,
        layer_idx: int,
        seq_ids: torch.Tensor,  # [T]
        positions: torch.Tensor,  # [T]
        block_tables: torch.Tensor,  # [n_seqs, max_blocks]
    ) -> torch.Tensor:
        """按 (seq, 绝对位置) 收集 KV 行 → [T, kv_cache_dim]。"""
        kv = self.layer_view(layer_idx)
        bs = self.cache_cfg.block_size
        blocks = block_tables[seq_ids, torch.div(positions, bs, rounding_mode="floor")]
        slots = positions % bs
        return kv[blocks, slots]

    def gather_range_reference(
        self,
        layer_idx: int,
        seq_row: int,
        num_tokens: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        """收集某 seq 前 num_tokens 个位置的 KV → [num_tokens, kv_cache_dim]。

        decode 吸收式注意力的参考路径（真实 kernel 走分页直接计算）。
        """
        kv = self.layer_view(layer_idx)
        bs = self.cache_cfg.block_size
        positions = torch.arange(num_tokens, device=self.device)
        blocks = block_tables[seq_row, torch.div(positions, bs, rounding_mode="floor")]
        slots = positions % bs
        return kv[blocks, slots]


__all__ = ["CacheEngine"]

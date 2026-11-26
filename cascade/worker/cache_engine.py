"""KV cache engine: GPU-side paged KV tensors.

The layout math lives in `cascade.worker.kv_layout` (torch-free). This module
owns the actual device tensors and the reference scatter/gather paths used
until (and alongside) the Triton kernels land. Block ids are shared across
layers — a block table row addresses `[layer, block, slot, :]`.

两种精度布局（`CacheConfig.kv_dtype`）：

- ``"bf16"``：单张量 `[L, n+1, bs, kv_lora+rope]`（c‖k_rope），默认路径。
- ``"fp8"``：三张量——latent c 存 FP8(e4m3)、k_rope 存 bf16、每 token 一个
  fp32 absmax/448 scale（DeepSeek-V3 生产同构）。KV 字节/token/层
  1152 → 644（≈1.79×），32 GB 卡的可容纳 block 数与并发随之提升。
  精度门槛由 e2e teacher-forcing 对拍把关（见 tests/e2e）。
"""

from __future__ import annotations

import torch

from cascade.common.logger import get_logger
from cascade.config import CacheConfig, ModelConfig
from cascade.worker.kv_layout import FP8_MAX, kv_shape, kv_shapes_fp8

_log = get_logger("worker.cache_engine")


class CacheEngine:
    __slots__ = (
        "model_cfg",
        "cache_cfg",
        "num_blocks",
        "dtype",
        "kv_dtype",
        "device",
        "kv_cache",
        "rope_cache",
        "scale_cache",
    )

    def __init__(
        self,
        model_cfg: ModelConfig,
        cache_cfg: CacheConfig,
        num_blocks: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: str | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.cache_cfg = cache_cfg
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.kv_dtype = kv_dtype or cache_cfg.kv_dtype
        assert self.kv_dtype in ("bf16", "fp8"), self.kv_dtype
        self.device = device
        if self.kv_dtype == "fp8":
            c_shape, r_shape, s_shape = kv_shapes_fp8(model_cfg, num_blocks, cache_cfg.block_size)
            # 单块扁平分配（DeepInfer cuMemAlloc 直分配的简化版；无 cuMem* 依赖）
            self.kv_cache = torch.zeros(c_shape, dtype=torch.float8_e4m3fn, device=device)
            self.rope_cache = torch.zeros(r_shape, dtype=dtype, device=device)
            self.scale_cache = torch.zeros(s_shape, dtype=torch.float32, device=device)
            total = (
                self.kv_cache.numel() * self.kv_cache.element_size()
                + self.rope_cache.numel() * self.rope_cache.element_size()
                + self.scale_cache.numel() * self.scale_cache.element_size()
            )
        else:
            shape = kv_shape(model_cfg, num_blocks, cache_cfg.block_size)
            self.kv_cache = torch.zeros(shape, dtype=dtype, device=device)
            self.rope_cache = None
            self.scale_cache = None
            total = self.kv_cache.numel() * self.kv_cache.element_size()
        _log.info(
            "KV cache allocated (%s): %s blocks × %d tokens/block, %.2f GiB",
            self.kv_dtype,
            num_blocks,
            cache_cfg.block_size,
            total / 2**30,
        )

    def layer_view(self, layer_idx: int) -> torch.Tensor:
        """第 layer_idx 层的 bf16 视图: [num_blocks + 1, block_size, kv_cache_dim]。

        fp8 模式下返回 **latent c 视图**（[.., kv_lora]）；需要 rope/scale 时
        用 `layer_views`。行 0 是保留的空块（BlockManager 的 block id 为
        1..num_blocks）。
        """
        return self.kv_cache[layer_idx]

    def layer_views(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """→ (latent/bf16 全量视图, rope 视图或 None, scale 视图或 None)。"""
        if self.kv_dtype == "fp8":
            return (
                self.kv_cache[layer_idx],
                self.rope_cache[layer_idx],  # type: ignore[index]
                self.scale_cache[layer_idx],  # type: ignore[index]
            )
        return self.kv_cache[layer_idx], None, None

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
        if self.kv_dtype == "fp8":
            from cascade.model_executor.triton_ops.store_paged_kv import store_paged_kv_fp8

            c, rope, scale = self.layer_views(layer_idx)
            assert rope is not None and scale is not None
            store_paged_kv_fp8(c, rope, scale, latent, seq_ids, positions, block_tables)
            return
        from cascade.model_executor.triton_ops.store_paged_kv import store_paged_kv

        kv = self.layer_view(layer_idx)
        store_paged_kv(kv, latent.to(kv.dtype), seq_ids, positions, block_tables)

    def _gather_rows(self, layer_idx: int, blocks: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """按 (block, slot) 索引取 KV 行 → [T, kv_cache_dim] 模型 dtype。"""
        if self.kv_dtype == "fp8":
            # CPU 的 fp8 高级索引未实现（index_cpu not implemented）——先按
            # int8 视图 gather 再 view 回 fp8，设备无关且不复制整层。
            c = (
                self.kv_cache[layer_idx]  # type: ignore[index]
                .view(torch.int8)[blocks, slots]
                .view(torch.float8_e4m3fn)
                .float()
            )
            sc = self.scale_cache[layer_idx][blocks, slots]  # type: ignore[index]
            groups = sc.shape[-1]
            # 分组 scale 展开回逐维（V2-Lite：4 组 × 128）
            c = c * sc.repeat_interleave(c.shape[-1] // groups, dim=-1)
            rope = self.rope_cache[layer_idx][blocks, slots].float()  # type: ignore[index]
            return torch.cat((c, rope), dim=-1).to(self.dtype)
        return self.kv_cache[layer_idx][blocks, slots]

    def gather_positions_reference(
        self,
        layer_idx: int,
        seq_ids: torch.Tensor,  # [T]
        positions: torch.Tensor,  # [T]
        block_tables: torch.Tensor,  # [n_seqs, max_blocks]
    ) -> torch.Tensor:
        """按 (seq, 绝对位置) 收集 KV 行 → [T, kv_cache_dim]。"""
        bs = self.cache_cfg.block_size
        blocks = block_tables[seq_ids, torch.div(positions, bs, rounding_mode="floor")]
        slots = positions % bs
        return self._gather_rows(layer_idx, blocks, slots)

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
        bs = self.cache_cfg.block_size
        positions = torch.arange(num_tokens, device=self.device)
        blocks = block_tables[seq_row, torch.div(positions, bs, rounding_mode="floor")]
        slots = positions % bs
        return self._gather_rows(layer_idx, blocks, slots)


__all__ = ["CacheEngine", "FP8_MAX"]
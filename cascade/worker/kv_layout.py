"""Paged-KV layout math — pure functions, no torch (Windows-testable).

MLA latent KV layout: each token occupies `kv_cache_dim = kv_lora_rank +
qk_rope_head_dim` values per layer (the normed latent c ‖ rope k). A *block*
covers `block_size` tokens of **one sequence position range across all
layers**: block id b addresses `[layer, b, slot, :dim]` — the block table is
shared by all layers (DeepInfer/vLLM semantics).
"""

from __future__ import annotations

from cascade.config import ModelConfig

#: bf16/fp16 每元素 2 字节；FP8 变体（interleaved [data|scale]）Phase 6 引入
DTYPE_SIZES = {"bf16": 2, "fp16": 2, "fp8": 1}


def kv_bytes_per_token(model_cfg: ModelConfig, dtype: str = "bf16") -> int:
    """单 token 单层的 KV 字节数。"""
    return model_cfg.kv_cache_dim * DTYPE_SIZES[dtype]


def kv_bytes_per_block(model_cfg: ModelConfig, block_size: int, dtype: str = "bf16") -> int:
    """单 block 单层的字节数（block = block_size 个 token 的该层 KV）。"""
    return kv_bytes_per_token(model_cfg, dtype) * block_size


def kv_total_bytes(model_cfg: ModelConfig, num_blocks: int, block_size: int, dtype: str = "bf16") -> int:
    """整个 KV 池的字节数（所有层 × (num_blocks + 1) 个 slot）。

    +1：block id 0 是保留的空块（BlockManager 的 block id 为 1..num_blocks），
    KV 张量必须能容纳 id = num_blocks。
    """
    return kv_bytes_per_block(model_cfg, block_size, dtype) * model_cfg.num_layers * (num_blocks + 1)


def derive_num_blocks(
    available_bytes: int,
    model_cfg: ModelConfig,
    block_size: int,
    dtype: str = "bf16",
) -> int:
    """给定可用显存预算，求可容纳的 block 数（向下取整，至少 1）。"""
    per_block_all_layers = kv_bytes_per_block(model_cfg, block_size, dtype) * model_cfg.num_layers
    assert per_block_all_layers > 0
    return max(1, available_bytes // per_block_all_layers)


def kv_shape(model_cfg: ModelConfig, num_blocks: int, block_size: int) -> tuple[int, int, int, int]:
    """KV 张量形状: [num_layers, num_blocks + 1, block_size, kv_cache_dim]。

    行 0 是保留的空块：BlockManager 分配的 block id 是 1..num_blocks，
    若只分配 num_blocks 行，id = num_blocks 会越界（explicit 路径
    PyTorch 索引直接 assert；absorbed kernel 手写地址则静默越界读写）。
    """
    return (model_cfg.num_layers, num_blocks + 1, block_size, model_cfg.kv_cache_dim)


__all__ = [
    "DTYPE_SIZES",
    "kv_bytes_per_token",
    "kv_bytes_per_block",
    "kv_total_bytes",
    "derive_num_blocks",
    "kv_shape",
]

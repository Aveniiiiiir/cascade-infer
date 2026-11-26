"""Paged-KV layout math — pure functions, no torch (Windows-testable).

MLA latent KV layout: each token occupies `kv_cache_dim = kv_lora_rank +
qk_rope_head_dim` values per layer (the normed latent c ‖ rope k). A *block*
covers `block_size` tokens of **one sequence position range across all
layers**: block id b addresses `[layer, b, slot, :dim]` — the block table is
shared by all layers (DeepInfer/vLLM semantics).
"""

from __future__ import annotations

from cascade.config import ModelConfig

#: bf16/fp16 每元素 2 字节；fp8 = latent c 1 B + rope bf16 2 B + 分组 scale
DTYPE_SIZES = {"bf16": 2, "fp16": 2, "fp8": 1}

#: FP8(e4m3) 的最大表示值（量化 scale = absmax / 448）
FP8_MAX = 448.0

#: FP8 KV 的分组 scale 粒度（DeepSeek-V3 生产格式：每 128 维一个 fp32 scale）
FP8_SCALE_GROUP = 128


def fp8_group_size(kv_lora_rank: int) -> int:
    """每组维度：128（512 等 128 的倍数）或整段（tiny 模型 <128）。"""
    return FP8_SCALE_GROUP if kv_lora_rank % FP8_SCALE_GROUP == 0 else kv_lora_rank


def fp8_num_groups(model_cfg: ModelConfig) -> int:
    return model_cfg.kv_lora_rank // fp8_group_size(model_cfg.kv_lora_rank)


def kv_bytes_per_token(model_cfg: ModelConfig, dtype: str = "bf16") -> int:
    """单 token 单层的 KV 字节数。

    fp8 混合布局：c(kv_lora) 1 B + k_rope 2 B + 每 128 维一个 fp32 scale。
    V2-Lite：512 + 128 + 16 = 656 B（bf16 为 1152 B，≈1.76×）。
    """
    if dtype == "fp8":
        groups = fp8_num_groups(model_cfg)
        return (
            model_cfg.kv_lora_rank
            + model_cfg.qk_rope_head_dim * 2
            + groups * 4
        )
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
    """给定可用显存预算，求可容纳的 block 数（向下取整）。

    实际张量是 (n+1) 行（行 0 保留），因此预算也要按 (n+1) 个 slot 计：
    n = available // per_block - 1。返回 0 表示连「保留行 + 1 个可用块」
    都放不下（调用方应显式失败，而不是静默超预算）。
    """
    per_block_all_layers = kv_bytes_per_block(model_cfg, block_size, dtype) * model_cfg.num_layers
    assert per_block_all_layers > 0
    return max(0, available_bytes // per_block_all_layers - 1)


def kv_shape(model_cfg: ModelConfig, num_blocks: int, block_size: int) -> tuple[int, int, int, int]:
    """KV 张量形状: [num_layers, num_blocks + 1, block_size, kv_cache_dim]。

    行 0 是保留的空块：BlockManager 分配的 block id 是 1..num_blocks，
    若只分配 num_blocks 行，id = num_blocks 会越界（explicit 路径
    PyTorch 索引直接 assert；absorbed kernel 手写地址则静默越界读写）。
    """
    return (model_cfg.num_layers, num_blocks + 1, block_size, model_cfg.kv_cache_dim)


def kv_shapes_fp8(
    model_cfg: ModelConfig, num_blocks: int, block_size: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """FP8 混合布局的三张量形状：(c, rope, scale)。

    c:     [L, n+1, bs, kv_lora]          float8_e4m3fn
    rope:  [L, n+1, bs, qk_rope]          bf16
    scale: [L, n+1, bs, num_groups]       fp32（每 128 维一个 absmax/448）
    """
    head = (model_cfg.num_layers, num_blocks + 1, block_size)
    return (
        (*head, model_cfg.kv_lora_rank),
        (*head, model_cfg.qk_rope_head_dim),
        (*head, fp8_num_groups(model_cfg)),
    )


__all__ = [
    "DTYPE_SIZES",
    "FP8_MAX",
    "FP8_SCALE_GROUP",
    "fp8_group_size",
    "fp8_num_groups",
    "kv_bytes_per_token",
    "kv_bytes_per_block",
    "kv_total_bytes",
    "derive_num_blocks",
    "kv_shape",
    "kv_shapes_fp8",
]

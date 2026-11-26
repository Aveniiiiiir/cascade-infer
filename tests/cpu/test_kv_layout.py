"""KV layout 纯函数测试（无 torch，Windows 可跑）。"""

from __future__ import annotations

from cascade.config import ModelConfig
from cascade.worker.kv_layout import (
    derive_num_blocks,
    kv_bytes_per_block,
    kv_bytes_per_token,
    kv_shape,
    kv_shapes_fp8,
    kv_total_bytes,
)


def make_cfg(num_layers: int = 27, kv_lora: int = 512, rope: int = 64) -> ModelConfig:
    return ModelConfig(num_layers=num_layers, kv_lora_rank=kv_lora, qk_rope_head_dim=rope)


def test_bytes_per_token() -> None:
    # V2-Lite: 576 dim × bf16(2B) = 1152 B/token/层
    assert kv_bytes_per_token(make_cfg()) == 576 * 2


def test_bytes_per_block() -> None:
    cfg = make_cfg()
    assert kv_bytes_per_block(cfg, block_size=64) == 576 * 2 * 64


def test_total_bytes_and_shape() -> None:
    """block id 0 保留 → 张量行数 = num_blocks + 1（id 最大值是 num_blocks）。"""
    cfg = make_cfg(num_layers=27)
    blocks = 100
    assert kv_total_bytes(cfg, blocks, 64) == 576 * 2 * 64 * 27 * 101
    assert kv_shape(cfg, blocks, 64) == (27, 101, 64, 576)


def test_derive_num_blocks() -> None:
    cfg = make_cfg(num_layers=27)
    per_block = 576 * 2 * 64 * 27  # 一个 block 全部 27 层的字节
    # 10 GiB 预算（按 (n+1) 个 slot 计）
    n = derive_num_blocks(10 * 2**30, cfg, 64)
    assert n == (10 * 2**30) // per_block - 1
    # 不足「保留行 + 1 个可用块」→ 0（调用方显式失败，而非超预算）
    assert derive_num_blocks(1024, cfg, 64) == 0
    # 恰好容纳 1 个可用块（2 个 slot）
    assert derive_num_blocks(2 * per_block, cfg, 64) == 1
    assert derive_num_blocks(2 * per_block - 1, cfg, 64) == 0


def test_v2_lite_realistic_footprint() -> None:
    """V2-Lite 尺寸直觉检查：10GiB KV 预算的 block 数与 token 容量。

    每 token 全 27 层 = 576×2×27 ≈ 31 KB（bf16）→ 10 GiB ≈ 5.4k block ≈ 34.5 万 token
    位置（MLA 压缩后 V2-Lite 的真实量级，bf16）。
    """
    cfg = make_cfg(num_layers=27)
    n = derive_num_blocks(10 * 2**30, cfg, 64)
    tokens = n * 64
    assert 5_000 < n < 6_000
    assert 320_000 < tokens < 360_000


def test_fp8_mixed_layout_bytes() -> None:
    """fp8：c 512 + rope 128 + 4 组×4B scale = 656 B/token/层（V3 生产格式）。"""
    cfg = make_cfg()
    assert kv_bytes_per_token(cfg, "fp8") == 656
    assert kv_bytes_per_block(cfg, 64, "fp8") == 656 * 64
    # 压缩比 656/1152 ≈ 0.569
    assert kv_bytes_per_token(cfg, "fp8") / kv_bytes_per_token(cfg, "bf16") < 0.57


def test_fp8_shapes() -> None:
    cfg = make_cfg(num_layers=27)
    c, r, s = kv_shapes_fp8(cfg, 100, 64)
    assert c == (27, 101, 64, 512)
    assert r == (27, 101, 64, 64)
    assert s == (27, 101, 64, 4)  # 每 128 维一个 scale


def test_fp8_derive_num_blocks() -> None:
    cfg = make_cfg(num_layers=27)
    per_block_fp8 = 656 * 64 * 27
    n = derive_num_blocks(10 * 2**30, cfg, 64, dtype="fp8")
    assert n == (10 * 2**30) // per_block_fp8 - 1
    n_bf16 = derive_num_blocks(10 * 2**30, cfg, 64)
    assert n > n_bf16  # fp8 同样预算容纳更多 block

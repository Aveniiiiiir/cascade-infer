"""KV layout 纯函数测试（无 torch，Windows 可跑）。"""

from __future__ import annotations

from cascade.config import ModelConfig
from cascade.worker.kv_layout import (
    derive_num_blocks,
    kv_bytes_per_block,
    kv_bytes_per_token,
    kv_shape,
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
    cfg = make_cfg(num_layers=27)
    blocks = 100
    assert kv_total_bytes(cfg, blocks, 64) == 576 * 2 * 64 * 27 * 100
    assert kv_shape(cfg, blocks, 64) == (27, 100, 64, 576)


def test_derive_num_blocks() -> None:
    cfg = make_cfg(num_layers=27)
    per_block = 576 * 2 * 64 * 27  # 一个 block 全部 27 层的字节
    # 10 GiB 预算
    n = derive_num_blocks(10 * 2**30, cfg, 64)
    assert n == (10 * 2**30) // per_block
    # 不足一个 block → 至少 1
    assert derive_num_blocks(1024, cfg, 64) == 1


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

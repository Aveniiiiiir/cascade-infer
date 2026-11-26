"""FP8 混合 KV 布局（c FP8 + rope bf16 + per-token scale）测试。

- store：量化落盘 → gather dequant ≈ 原值（e4m3 相对误差量级）
- decode / prefill kernel vs ref（ref 走同一 dequant 路径，测的是 kernel 数学）
- uniform_v（CUDA graph 多 token）与 segments 路径一致
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.paged_mla_decode_attn import (
    paged_mla_decode_attn,
    paged_mla_decode_attn_ref,
)
from cascade.model_executor.triton_ops.paged_mla_prefill_attn import (
    paged_mla_prefill_attn,
    paged_mla_prefill_attn_ref,
)
from cascade.model_executor.triton_ops.store_paged_kv import store_paged_kv_fp8

SCALE = (128 + 64) ** -0.5
H = 16
KV_LORA = 512
ROPE = 64
D = KV_LORA + ROPE
FP8_MAX = 448.0


def _quantize(kv_bf16: torch.Tensor):
    """bf16 [blocks, BS, D] → (c_fp8, rope_bf16, scale[blocks,BS,4])。

    每 128 维一组 absmax（DeepSeek-V3 生产格式）。
    """
    c = kv_bf16[..., :KV_LORA].float().view(*kv_bf16.shape[:2], 4, 128)
    sc = (c.abs().amax(dim=-1) / FP8_MAX).clamp_min(1e-12)  # [B, BS, 4]
    cq = (
        (c / sc.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(*kv_bf16.shape[:2], KV_LORA)
    )
    return cq, kv_bf16[..., KV_LORA:].contiguous(), sc


def _make_case(device: str, n_rows: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    kv = (torch.randn(n_rows, 64, D, generator=g) * 0.05).to(torch.bfloat16).to(device)
    cq, rope, sc = _quantize(kv)
    q = (torch.randn(4, H, D, generator=g) * 0.05).to(torch.bfloat16).to(device)
    return q, cq, rope, sc, kv


def test_store_paged_kv_fp8_roundtrip() -> None:
    """CPU ref 路径：量化写盘 → dequant gather ≈ 原 latent（e4m3 容差）。"""
    g = torch.Generator().manual_seed(1)
    BS = 8
    nb = 5
    c = torch.zeros(nb, BS, KV_LORA, dtype=torch.float8_e4m3fn)
    rope = torch.zeros(nb, BS, ROPE, dtype=torch.bfloat16)
    sc = torch.zeros(nb, BS, 4, dtype=torch.float32)
    T = 9
    latent = (torch.randn(T, D, generator=g) * 0.1).to(torch.bfloat16)
    seq_ids = torch.zeros(T, dtype=torch.int64)
    positions = torch.arange(T, dtype=torch.int64)
    bt = torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.int64)
    store_paged_kv_fp8(c, rope, sc, latent, seq_ids, positions, bt)
    blocks = bt[seq_ids, positions // BS]
    slots = positions % BS
    sc_full = sc[blocks, slots].repeat_interleave(KV_LORA // 4, dim=-1)
    got_c = (
        c.view(torch.int8)[blocks, slots].view(torch.float8_e4m3fn).float() * sc_full
    )
    got_rope = rope[blocks, slots].float()
    torch.testing.assert_close(got_c, latent[:, :KV_LORA].float(), rtol=8e-2, atol=2e-2)
    torch.testing.assert_close(got_rope, latent[:, KV_LORA:].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_decode_fp8_kernel_matches_ref() -> None:
    q, cq, rope, sc, _kv = _make_case("cuda", n_rows=140, seed=2)
    N = q.shape[0]
    BS = cq.shape[1]
    seqlens = [1, 63, 65, 1000][:N]
    max_blocks = (max(seqlens) + BS - 1) // BS
    bt = torch.zeros(N, max_blocks, dtype=torch.int64, device="cuda")
    for i in range(N):
        need = (seqlens[i] + BS - 1) // BS
        bt[i, :need] = torch.arange(1 + i * max_blocks, 1 + i * max_blocks + need, device="cuda")
    # 保证 block id < cq.shape[0]
    assert int(bt.max()) < cq.shape[0]
    sl = torch.tensor(seqlens, dtype=torch.int64, device="cuda")
    got = paged_mla_decode_attn(
        q, cq, bt, sl, SCALE, KV_LORA, rope=rope, kv_scale=sc
    )
    ref = paged_mla_decode_attn_ref(
        q, cq, bt, sl, SCALE, KV_LORA, rope, sc
    )
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_prefill_fp8_kernel_matches_ref() -> None:
    g = torch.Generator().manual_seed(3)
    BS = 64
    T = 64
    kv = (torch.randn(8, BS, D, generator=g) * 0.05).to(torch.bfloat16).to("cuda")
    cq, rope, sc = _quantize(kv)
    q = (torch.randn(T, H, D, generator=g) * 0.05).to(torch.bfloat16).to("cuda")
    bt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int64, device="cuda")
    seqlens = torch.tensor([256, 130], dtype=torch.int64, device="cuda")
    segments = [(0, 0, 40), (1, 0, 24)]
    got = paged_mla_prefill_attn(
        q, cq, bt, seqlens, segments, SCALE, KV_LORA, rope=rope, kv_scale=sc
    )
    ref = paged_mla_prefill_attn_ref(
        q, cq, bt, seqlens, segments, SCALE, KV_LORA, rope, sc
    )
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_prefill_fp8_uniform_v_matches_segments() -> None:
    """uniform_v（图捕获路径）与显式 segments 结果一致。"""
    g = torch.Generator().manual_seed(4)
    BS = 64
    n, v = 3, 4
    kv = (torch.randn(8, BS, D, generator=g) * 0.05).to(torch.bfloat16).to("cuda")
    cq, rope, sc = _quantize(kv)
    q = (torch.randn(n * v, H, D, generator=g) * 0.05).to(torch.bfloat16).to("cuda")
    # uniform 契约：每段是序列的**尾部** v 个 token（seqlens = pos_start + v）
    bt = torch.tensor([[1], [2], [3]], dtype=torch.int64, device="cuda")
    seqlens = torch.tensor([24, 24, 24], dtype=torch.int64, device="cuda")
    segs = [(i, 20, v) for i in range(n)]
    a = paged_mla_prefill_attn(
        q, cq, bt, seqlens, segs, SCALE, KV_LORA, rope=rope, kv_scale=sc
    )
    b = paged_mla_prefill_attn(
        q, cq, bt, seqlens, [], SCALE, KV_LORA, uniform_v=v,
        rope=rope, kv_scale=sc,
    )
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
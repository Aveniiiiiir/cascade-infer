"""分页 MLA attention kernel 测试：decode split-K / prefill varlen causal。

- CUDA + bf16：kernel vs ref（容差对应 bf16 dot 概率路径）
- CPU / fp32：ref vs 独立手工实现（紧容差，防 ref 自身出错）
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

SCALE = (128 + 64) ** -0.5  # 192^-0.5
H = 16
KV_LORA = 512
ROPE = 64
D = KV_LORA + ROPE


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_decode_case(device: str, dtype: torch.dtype, seqlens: list[int], seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    N = len(seqlens)
    BS = 64
    max_blocks = max((s + BS - 1) // BS for s in seqlens)
    total_blocks = N * max_blocks + 1
    kv = (torch.randn(total_blocks, BS, D, generator=g) * 0.05).to(dtype).to(device)
    q = (torch.randn(N, H, D, generator=g) * 0.05).to(dtype).to(device)
    bt = torch.zeros(N, max_blocks, dtype=torch.int64, device=device)
    for i in range(N):
        need = (seqlens[i] + BS - 1) // BS
        bt[i, :need] = torch.arange(i * max_blocks + 1, i * max_blocks + need + 1, device=device)
    sl = torch.tensor(seqlens, dtype=torch.int64, device=device)
    return q, kv, bt, sl


def _manual_decode(q, kv, bt, sl, scale, kv_lora):
    """独立手工实现（oracle 的 oracle，纯 fp32）。"""
    N, H, _D = q.shape
    BS = kv.shape[1]
    out = torch.empty(N, H, kv_lora, dtype=torch.float32, device=q.device)
    for n in range(N):
        P = int(sl[n])
        nb = (P + BS - 1) // BS
        rows = kv[bt[n, :nb].long()].reshape(-1, _D)[:P].float()
        scores = torch.zeros(H, P, device=q.device)
        for h in range(H):
            for p in range(P):
                scores[h, p] = torch.dot(q[n, h].float(), rows[p]) * scale
        w = torch.softmax(scores, dim=-1)
        for h in range(H):
            out[n, h] = (w[h, :, None] * rows[:, :kv_lora]).sum(dim=0)
    return out


@pytest.mark.parametrize("device", _devices())
def test_decode_ref_matches_manual(device: str) -> None:
    seqlens = [1, 63, 64, 65, 130]
    q, kv, bt, sl = _make_decode_case(device, torch.float32, seqlens)
    ref = paged_mla_decode_attn_ref(q, kv, bt, sl, SCALE, KV_LORA)
    manual = _manual_decode(q, kv, bt, sl, SCALE, KV_LORA)
    torch.testing.assert_close(ref, manual, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_decode_kernel_matches_ref() -> None:
    seqlens = [1, 31, 63, 64, 65, 200, 500, 1000]
    q, kv, bt, sl = _make_decode_case("cuda", torch.bfloat16, seqlens, seed=3)
    got = paged_mla_decode_attn(q, kv, bt, sl, SCALE, KV_LORA)
    ref = paged_mla_decode_attn_ref(q, kv, bt, sl, SCALE, KV_LORA)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_decode_kernel_num_splits_variants() -> None:
    """显式 num_splits=1（无分片）与默认分片结果一致。"""
    seqlens = [100, 333, 700]
    q, kv, bt, sl = _make_decode_case("cuda", torch.bfloat16, seqlens, seed=4)
    a = paged_mla_decode_attn(q, kv, bt, sl, SCALE, KV_LORA, num_splits=1)
    b = paged_mla_decode_attn(q, kv, bt, sl, SCALE, KV_LORA)  # 默认多分片
    ref = paged_mla_decode_attn_ref(q, kv, bt, sl, SCALE, KV_LORA)
    torch.testing.assert_close(a, ref, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(b, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("device", _devices())
def test_decode_wrapper_cpu_fallback(device: str) -> None:
    """wrapper 在无 triton/fp32 输入时走 ref（结果与直调 ref 一致）。"""
    seqlens = [5, 64, 129]
    q, kv, bt, sl = _make_decode_case(device, torch.float32, seqlens, seed=5)
    got = paged_mla_decode_attn(q, kv, bt, sl, SCALE, KV_LORA)
    ref = paged_mla_decode_attn_ref(q, kv, bt, sl, SCALE, KV_LORA)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------- prefill


def _make_prefill_case(
    device: str, dtype: torch.dtype, segments: list[tuple[int, int, int]], seed: int = 0
):
    g = torch.Generator().manual_seed(seed)
    T = sum(seg[2] for seg in segments)
    BS = 64
    n_seqs = max(s[0] for s in segments) + 1
    max_len = max(pos + ln for _, pos, ln in segments)
    max_blocks = (max_len + BS - 1) // BS
    kv = (torch.randn(n_seqs * max_blocks + 1, BS, D, generator=g) * 0.05).to(dtype).to(device)
    q = (torch.randn(T, H, D, generator=g) * 0.05).to(dtype).to(device)
    bt = torch.zeros(n_seqs, max_blocks, dtype=torch.int64, device=device)
    for i in range(n_seqs):
        bt[i, :max_blocks] = torch.arange(i * max_blocks + 1, (i + 1) * max_blocks + 1, device=device)
    seqlens = torch.tensor([max_len] * n_seqs, dtype=torch.int64, device=device)
    return q, kv, bt, seqlens


def _manual_prefill(q, kv, bt, seqlens, segments, scale, kv_lora):
    T, H, D = q.shape
    BS = kv.shape[1]
    out = torch.empty(T, H, kv_lora, dtype=torch.float32, device=q.device)
    base = 0
    for row, pos_start, seg_len in segments:
        P = int(seqlens[row])
        nb = (P + BS - 1) // BS
        rows = kv[bt[row, :nb].long()].reshape(-1, D)[:P].float()
        for m in range(seg_len):
            pos_q = pos_start + m
            scores = torch.empty(H, P, device=q.device)
            for h in range(H):
                for p in range(P):
                    scores[h, p] = torch.dot(q[base + m, h].float(), rows[p]) * scale if p <= pos_q else float("-inf")
            w = torch.softmax(scores, dim=-1)
            for h in range(H):
                out[base + m, h] = (w[h, :, None] * rows[:, :kv_lora]).sum(dim=0)
        base += seg_len
    return out


@pytest.mark.parametrize("device", _devices())
def test_prefill_ref_matches_manual(device: str) -> None:
    # 混合段：完整 prompt、chunked 中段（pos_start>0）、跨 block 边界长度
    segments = [(0, 0, 33), (1, 0, 65), (0, 33, 70), (2, 0, 1)]
    q, kv, bt, seqlens = _make_prefill_case(device, torch.float32, segments, seed=6)
    ref = paged_mla_prefill_attn_ref(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    manual = _manual_prefill(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    torch.testing.assert_close(ref, manual, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_prefill_kernel_matches_ref() -> None:
    segments = [(0, 0, 33), (1, 0, 130), (0, 33, 64), (2, 0, 1), (3, 0, 200)]
    q, kv, bt, seqlens = _make_prefill_case("cuda", torch.bfloat16, segments, seed=7)
    got = paged_mla_prefill_attn(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    ref = paged_mla_prefill_attn_ref(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_prefill_single_long_segment() -> None:
    """单一长段（512 token = 32 个 q 块 × 多 kv 块）。"""
    segments = [(0, 0, 512)]
    q, kv, bt, seqlens = _make_prefill_case("cuda", torch.bfloat16, segments, seed=8)
    got = paged_mla_prefill_attn(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    ref = paged_mla_prefill_attn_ref(q, kv, bt, seqlens, segments, SCALE, KV_LORA)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)

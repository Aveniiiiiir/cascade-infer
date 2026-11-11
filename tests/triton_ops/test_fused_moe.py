"""fused_moe：kernel vs ref 对拍 + 逐 token 手工 oracle + 确定性。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.fused_moe import (
    fused_experts,
    fused_experts_ref,
)


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_case(device: str, dtype: torch.dtype, T: int, K: int, E: int, N: int, H: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g).to(dtype).to(device)
    w1 = (torch.randn(E, N, H, generator=g) * 0.02).to(dtype).to(device)
    w3 = (torch.randn(E, N, H, generator=g) * 0.02).to(dtype).to(device)
    w2 = (torch.randn(E, H, N, generator=g) * 0.02).to(dtype).to(device)
    topk_ids = torch.randint(0, E, (T, K), generator=g).to(torch.int32).to(device)
    topk_weight = torch.rand(T, K, generator=g).to(torch.float32).to(device) * 2.5
    return x, w1, w3, w2, topk_ids, topk_weight


def _manual_per_token(x, w1, w3, w2, topk_ids, topk_weight):
    """独立逐 token oracle（fp32）。"""
    T, H = x.shape
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for t in range(T):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            h1 = x[t].float() @ w1[e].float().T
            h3 = x[t].float() @ w3[e].float().T
            act = torch.nn.functional.silu(h1) * h3
            out[t] += float(topk_weight[t, k]) * (act @ w2[e].float().T)
    return out


@pytest.mark.parametrize("device", _devices())
def test_matches_manual_per_token(device: str) -> None:
    """wrapper（cpu→ref 路径）vs 独立逐 token 手工实现（fp32 紧容差）。"""
    x, w1, w3, w2, ids, wts = _make_case(device, torch.float32, T=16, K=3, E=4, N=32, H=64, seed=1)
    got = fused_experts(x, w1, w3, w2, wts, ids)
    ref = _manual_per_token(x, w1, w3, w2, ids, wts)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_kernel_matches_ref() -> None:
    x, w1, w3, w2, ids, wts = _make_case(
        "cuda", torch.bfloat16, T=128, K=6, E=8, N=1408, H=2048, seed=2
    )
    got = fused_experts(x, w1, w3, w2, wts, ids)
    ref = fused_experts_ref(x, w1, w3, w2, wts, ids)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_kernel_deterministic() -> None:
    x, w1, w3, w2, ids, wts = _make_case("cuda", torch.bfloat16, T=64, K=4, E=16, N=256, H=512, seed=3)
    a = fused_experts(x, w1, w3, w2, wts, ids)
    b = fused_experts(x, w1, w3, w2, wts, ids)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel 需要 CUDA")
def test_kernel_row_subset_invariance() -> None:
    """行子集不变：batch 组合不影响输出（批不变——无原子 + 固定归约序）。"""
    x, w1, w3, w2, ids, wts = _make_case("cuda", torch.bfloat16, T=64, K=4, E=8, N=128, H=256, seed=4)
    full = fused_experts(x, w1, w3, w2, wts, ids)
    g = torch.Generator().manual_seed(5)
    perm = torch.randperm(64, generator=g)[:32].to("cuda")
    sub = fused_experts(x[perm], w1, w3, w2, wts[perm], ids[perm])
    torch.testing.assert_close(sub, full[perm], rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("device", _devices())
def test_empty_tokens(device: str) -> None:
    x, w1, w3, w2, ids, wts = _make_case(device, torch.float32, T=4, K=2, E=4, N=16, H=32, seed=6)
    out = fused_experts(x[:0], w1, w3, w2, wts[:0], ids[:0])
    assert out.shape == (0, x.shape[1])

"""det_random_sampling：kernel vs ref bit 级对拍、零概率列不变式、批不变。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.det_random_sampling import (
    det_random_sampling,
    det_random_sampling_nextn,
)
from cascade.sampling.rng import random_sampling_ref_pure


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_probs(T: int, vocab: int, device: str, seed: int = 0, zero_cols: bool = True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, vocab, generator=g, dtype=torch.float32)
    if zero_cols:
        logits[:, ::5] = float("-inf")  # 零概率列
    probs = torch.softmax(logits, dim=-1)
    return probs.to(device).contiguous()


def _make_inputs(T: int, device: str, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    # 种子含 ≥2^63 的位模式（uint64 全域）
    seeds = torch.randint(-2**62, 2**62, (T,), generator=g, dtype=torch.int64).to(device)
    positions = torch.randint(0, 100_000, (T,), generator=g, dtype=torch.int64).to(device)
    return seeds, positions


@pytest.mark.parametrize("device", _devices())
def test_matches_pure_oracle(device: str) -> None:
    """wrapper（cuda→Triton kernel / cpu→torch ref）vs 纯 Python 公式 bit 级一致。"""
    T, vocab = 64, 1024
    probs = _make_probs(T, vocab, device)
    seeds, positions = _make_inputs(T, device)
    out = det_random_sampling(probs, seeds, positions)
    pure = random_sampling_ref_pure(
        [r.tolist() for r in probs.cpu()], seeds.cpu().tolist(), positions.cpu().tolist()
    )
    assert out.tolist() == pure


@pytest.mark.parametrize("device", _devices())
def test_zero_prob_never_sampled(device: str) -> None:
    T, vocab = 128, 512
    probs = _make_probs(T, vocab, device, seed=5)
    seeds, positions = _make_inputs(T, device, seed=6)
    out = det_random_sampling(probs, seeds, positions)
    for i in range(T):
        assert probs[i, int(out[i])] > 0, f"row {i}: 采到零概率列 {int(out[i])}"


@pytest.mark.parametrize("device", _devices())
def test_deterministic(device: str) -> None:
    T, vocab = 32, 256
    probs = _make_probs(T, vocab, device, seed=8)
    seeds, positions = _make_inputs(T, device, seed=9)
    a = det_random_sampling(probs, seeds, positions)
    b = det_random_sampling(probs, seeds, positions)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", _devices())
def test_row_order_invariance(device: str) -> None:
    """行置换后每行采样不变 —— 采样层批不变（kernel 由构造保证）。"""
    T, vocab = 96, 512
    probs = _make_probs(T, vocab, device, seed=10)
    seeds, positions = _make_inputs(T, device, seed=11)
    base = det_random_sampling(probs, seeds, positions)
    g = torch.Generator().manual_seed(12)
    perm = torch.randperm(T, generator=g).to(device)
    shuffled = det_random_sampling(probs[perm], seeds[perm], positions[perm])
    assert torch.equal(shuffled, base[perm])


@pytest.mark.parametrize("device", _devices())
def test_nextn_stream_independent(device: str) -> None:
    T, vocab = 64, 256
    probs = _make_probs(T, vocab, device, seed=13)
    seeds, positions = _make_inputs(T, device, seed=14)
    plain = det_random_sampling(probs, seeds, positions)
    nextn = det_random_sampling_nextn(probs, seeds, positions)
    diff = (plain != nextn).sum().item()
    # 独立随机流：几乎必然不同（64 行全同概率 ≈ vocab^-63）
    assert diff > T // 2, f"draft 流与普通流差异过小（{diff}/{T}）"


@pytest.mark.parametrize("device", _devices())
def test_uniformity_distribution(device: str) -> None:
    """大样本下的均匀性：各列命中频率与概率成比例（卡方粗检）。"""
    T, vocab = 4096, 64
    probs = torch.full((T, vocab), 1.0 / vocab, device=device)
    seeds = torch.arange(T, dtype=torch.int64, device=device)
    positions = torch.zeros(T, dtype=torch.int64, device=device)
    out = det_random_sampling(probs, seeds, positions)
    counts = torch.bincount(out.long(), minlength=vocab).float()
    expected = T / vocab
    sigma = (T * (1 / vocab) * (1 - 1 / vocab)) ** 0.5
    assert (counts - expected).abs().max() < 6 * sigma, (
        f"桶计数偏差过大: max={(counts - expected).abs().max():.0f} vs 6σ={6 * sigma:.0f}"
    )

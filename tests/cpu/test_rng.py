"""Deterministic RNG tests: golden vectors, batch-order independence,
zero-probability invariants, pure-Python ↔ torch bit-level parity.

These encode the DeepInfer scheme (random_sampling.py:11-84) that makes
sampling invariant to batch composition.
"""

from __future__ import annotations

import math

import pytest

from cascade.sampling.rng import (
    CASE_NEXTN,
    CASE_SAMPLE,
    CASE_VERIFY,
    derive_uniform,
    derive_uniform_nextn,
    random_sampling_ref_pure,
    sample_cdf_inverse,
    splitmix64_to_uniform,
)
from cascade.sampling_params import MTP_SEED_OFFSET

# --------------------------------------------------------------------- golden


def _splitmix64_reference(z: int) -> int:
    """测试内独立实现的 SplitMix64 finalizer（与被测代码无共享实现）。"""
    M = (1 << 64) - 1
    z = (z + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


@pytest.mark.parametrize(
    ("seed", "u_expected"),
    [
        (0, (_splitmix64_reference(0) >> 11) / (1 << 53)),
        (1, (_splitmix64_reference(1) >> 11) / (1 << 53)),
        (0xDEADBEEFCAFEBABE, (_splitmix64_reference(0xDEADBEEFCAFEBABE) >> 11) / (1 << 53)),
        ((1 << 64) - 1, (_splitmix64_reference((1 << 64) - 1) >> 11) / (1 << 53)),
    ],
)
def test_splitmix64_golden(seed: int, u_expected: float) -> None:
    assert splitmix64_to_uniform(seed) == u_expected


def test_derive_uniform_golden() -> None:
    """手工按公式重算 derive_uniform（含 uint64 回绕）。"""
    M = (1 << 64) - 1
    for seed, pos, case in [(12345, 0, 0), (7, 100, 1), (0xFFFFFFFFFFFFFFFF, 999, 2)]:
        seq = ((pos * 3 + case) * 0x9E3779B97F4A7C15) & M
        mixed = seq ^ ((seq << 13) & M)
        combined = (seed + mixed) & M
        expected = (_splitmix64_reference(combined) >> 11) / (1 << 53)
        assert derive_uniform(seed, pos, case) == expected


def test_derive_uniform_range() -> None:
    for seed in (0, 1, 2**63, 2**64 - 1):
        for pos in (0, 1, 127, 2**31):
            for case in (CASE_SAMPLE, CASE_NEXTN, CASE_VERIFY):
                u = derive_uniform(seed, pos, case)
                assert 0.0 <= u < 1.0


# --------------------------------------------------------------------- properties


def test_determinism() -> None:
    assert derive_uniform(42, 17, CASE_SAMPLE) == derive_uniform(42, 17, CASE_SAMPLE)


def test_streams_are_independent() -> None:
    """case 流 + draft 偏移产生不同的随机数序列。"""
    seed, pos = 999, 5
    assert derive_uniform(seed, pos, 0) != derive_uniform(seed, pos, 1) or seed == 0
    # nextn 流：偏移 + case=1
    u_nextn = derive_uniform_nextn(seed, pos)
    u_manual = derive_uniform((seed + MTP_SEED_OFFSET) & (2**64 - 1), pos, CASE_NEXTN)
    assert u_nextn == u_manual


def test_uniformity_buckets() -> None:
    """64 桶均匀性：每桶计数与期望在 5σ 内（1 万样本）。"""
    n, buckets = 10_000, 64
    counts = [0] * buckets
    for pos in range(n):
        u = derive_uniform(12345, pos, CASE_SAMPLE)
        counts[min(int(u * buckets), buckets - 1)] += 1
    expected = n / buckets
    sigma = math.sqrt(n * (1 / buckets) * (1 - 1 / buckets))
    for c in counts:
        assert abs(c - expected) < 5 * sigma, f"bucket count {c} vs {expected}±{5 * sigma:.1f}"


def test_cdf_inverse_basic() -> None:
    probs = [0.1, 0.2, 0.7]
    assert sample_cdf_inverse(probs, 0.0) == 0  # 第一个非零列
    assert sample_cdf_inverse(probs, 0.999) == 2
    assert sample_cdf_inverse(probs, 0.05) == 0
    assert sample_cdf_inverse(probs, 0.35) == 2  # 0.1+0.2=0.3 ≤ 0.35


def test_cdf_inverse_zero_prob_never_sampled() -> None:
    probs = [0.0, 0.5, 0.0, 0.5]
    for i in range(1000):
        u = i / 1000.0
        col = sample_cdf_inverse(probs, u)
        assert probs[col] > 0, f"zero-prob column {col} sampled at u={u}"


def test_cdf_inverse_all_zero_raises() -> None:
    with pytest.raises(ValueError):
        sample_cdf_inverse([0.0, 0.0], 0.5)


# --------------------------------------------------------------------- torch parity


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch 未安装")
def test_torch_ref_bit_identical() -> None:
    """纯 Python 规范实现与 torch 向量化 ref 逐行 bit 级一致。"""
    import torch

    from cascade.sampling.rng import random_sampling_ref

    torch.manual_seed(0)
    rows, vocab = 64, 1024
    probs = torch.rand(rows, vocab).double()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    # 零概率列注入
    probs[:, ::7] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    seeds = torch.randint(0, 2**63, (rows,), dtype=torch.int64)
    positions = torch.randint(0, 2**31, (rows,), dtype=torch.int64)

    got_torch = random_sampling_ref(probs, seeds, positions).tolist()
    got_pure = random_sampling_ref_pure(
        [r.tolist() for r in probs], seeds.tolist(), positions.tolist()
    )
    assert got_torch == got_pure

    # nextn 流（case=1 + 偏移）
    got_torch_n = random_sampling_ref(probs, seeds, positions, is_nextn_predict=True).tolist()
    got_pure_n = random_sampling_ref_pure(
        [r.tolist() for r in probs],
        [(s + MTP_SEED_OFFSET) % 2**64 for s in seeds.tolist()],
        positions.tolist(),
        case=CASE_NEXTN,
    )
    assert got_torch_n == got_pure_n


@pytest.mark.skipif(not _torch_available(), reason="torch 未安装")
def test_torch_ref_row_order_invariant() -> None:
    """行置换后同一行采样结果不变（采样层批不变的最直接表达）。"""
    import torch

    torch.manual_seed(1)
    rows, vocab = 128, 512
    probs = torch.rand(rows, vocab).double()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    seeds = torch.randint(0, 2**63, (rows,), dtype=torch.int64)
    positions = torch.arange(rows, dtype=torch.int64) * 37

    base = random_sampling_ref(probs, seeds, positions)
    perm = torch.randperm(rows)
    shuffled = random_sampling_ref(probs[perm], seeds[perm], positions[perm])
    assert torch.equal(shuffled, base[perm])


@pytest.mark.skipif(not _torch_available(), reason="torch 未安装")
def test_torch_ref_zero_prob_never() -> None:
    import torch

    torch.manual_seed(2)
    vocab = 256
    probs = torch.rand(512, vocab).double()
    probs[:, ::5] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    seeds = torch.randint(0, 2**63, (512,), dtype=torch.int64)
    positions = torch.arange(512, dtype=torch.int64)
    out = random_sampling_ref(probs, seeds, positions)
    for i in range(512):
        assert probs[i, out[i].item()] > 0, f"row {i}: sampled zero-prob column {out[i]}"

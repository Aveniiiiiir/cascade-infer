"""Deterministic sampling RNG — bit-level replica of DeepInfer's scheme.

Reference: `deepinfer/model_executor/layers/tilelang_ops/random_sampling.py`
(:11-84). The canonical derivation (pure Python, no torch) is:

    seq       = ((position * 3 + case) * 0x9E3779B97F4A7C15) mod 2^64
    combined  = (seq_seed + (seq ^ (seq << 13))) mod 2^64        (uint64 wrap)
    u         = splitmix64_to_uniform(combined)                   ∈ [0, 1)
    sample    = argmin{ i : CDF_i > u * CDF_total }               (fp64 逆 CDF)

case ∈ {0 = 普通采样, 1 = nextn/draft 预测, 2 = verify}；draft 流另加
`MTP_SEED_OFFSET`（DeepInfer fused_sample.py:15）。随机流只由
(seed, token 绝对位置, 用途) 决定 —— 与 batch 组合、行顺序、设备无关，
这就是采样层天然批不变的原因。

The torch-vectorized `*_ref` functions emulate uint64 arithmetic with int64
(the `i64/urshift` trick from DeepInfer) and are the oracle for the Triton
kernel in Phase 3.
"""

from __future__ import annotations

from typing import Sequence as SequenceABC

from cascade.sampling_params import MTP_SEED_OFFSET

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_SM_A, _SM_B = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB

CASE_SAMPLE = 0
CASE_NEXTN = 1
CASE_VERIFY = 2


# --------------------------------------------------------------------- pure python


def splitmix64_to_uniform(seed: int) -> float:
    """SplitMix64 finalizer → uniform float64 in [0, 1)（53-bit 精度）。"""
    z = (seed + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * _SM_A) & _MASK64
    z = ((z ^ (z >> 27)) * _SM_B) & _MASK64
    z ^= z >> 31
    return (z >> 11) * (2.0**-53)


def derive_uniform(seq_seed: int, position: int, case: int = CASE_SAMPLE) -> float:
    """(seed, 绝对位置, 用途流) → uniform [0,1)。bit 级复刻 DeepInfer。"""
    seq = ((position * 3 + case) * _GOLDEN) & _MASK64
    mixed = seq ^ ((seq << 13) & _MASK64)
    combined = (seq_seed + mixed) & _MASK64
    return splitmix64_to_uniform(combined)


def derive_uniform_nextn(seq_seed: int, position: int) -> float:
    """draft/nextn 流：独立偏移，保证 draft 与 target 不共享随机数。"""
    return derive_uniform((seq_seed + MTP_SEED_OFFSET) & _MASK64, position, CASE_NEXTN)


def sample_cdf_inverse(probs: SequenceABC[float], u: float) -> int:
    """fp64 累积 CDF 逆变换。零概率列永不命中（CDF 在该列上不前进）。"""
    total = 0.0
    for p in probs:
        total += p
    target = u * total
    cdf = 0.0
    for i, p in enumerate(probs):
        cdf += p
        if cdf > target:
            return i
    # u ∈ [0,1) 且 total>0 ⇒ 必然命中；防御性回退返回最后一个非零列
    for i in range(len(probs) - 1, -1, -1):
        if probs[i] > 0:
            return i
    raise ValueError("sample_cdf_inverse: all-zero probability row")


def random_sampling_ref_pure(
    probs: SequenceABC[SequenceABC[float]],
    seq_seeds: SequenceABC[int],
    sample_positions: SequenceABC[int],
    case: int = CASE_SAMPLE,
) -> list[int]:
    """纯 Python 规范实现（无 torch 依赖，Windows/任意平台可跑）。"""
    out: list[int] = []
    for row, seed, pos in zip(probs, seq_seeds, sample_positions):
        u = derive_uniform(seed, pos, case)
        out.append(sample_cdf_inverse(row, u))
    return out


# --------------------------------------------------------------------- torch ref


def _i64(x: int) -> int:
    """把 uint64 解释为 int64（torch int64 溢出按二补码回绕）。"""
    return x if x < 0x8000000000000000 else x - 0x10000000000000000


def _urshift(x, n: int):
    """int64 张量的逻辑右移（无符号语义）。"""
    shifted = x.div(1 << n, rounding_mode="floor")
    return shifted + (x < 0).to(x.dtype) * (1 << (64 - n))


def _hash_to_uniform(seed):
    """SplitMix64 of int64 tensor → float64 tensor in [0,1)。torch 版。"""
    import torch  # 局部导入：_hash_to_uniform 独立可用（_urshift 为鸭子类型）

    z = seed.to(torch.int64)
    z = z + _i64(_GOLDEN)
    z = (z ^ _urshift(z, 30)) * _i64(_SM_A)
    z = (z ^ _urshift(z, 27)) * _i64(_SM_B)
    z = z ^ _urshift(z, 31)
    return _urshift(z, 11).to(torch.float64) * (1.0 / (1 << 53))


def random_sampling_ref(
    probs,
    seq_seeds,
    sample_positions,
    is_nextn_predict: bool = False,
):
    """torch 向量化参考实现（设备无关，CPU 可跑）。

    bit 级对齐 DeepInfer random_sampling.py:70-84；Triton kernel 的
    oracle。is_nextn_predict=True 时用 draft 流（case=1 + MTP_SEED_OFFSET）。
    """
    import torch  # 局部导入：CPU 调度栈不依赖 torch

    sample_case = 1 if is_nextn_predict else 0
    seeds = seq_seeds.to(torch.int64)
    if is_nextn_predict:
        seeds = seeds + _i64(MTP_SEED_OFFSET)
    seq = (sample_positions.to(torch.int64) * 3 + sample_case) * _i64(_GOLDEN)
    combined = seeds + (seq ^ (seq * (1 << 13)))
    u = _hash_to_uniform(combined).unsqueeze(-1)
    cdf = probs.to(torch.float64).cumsum(dim=-1)
    target = u * cdf[:, -1:]
    mask = cdf > target
    out = mask.long().argmax(dim=-1)
    # 理论边界（概率行总和为次正规数时 mask 可能全 False，argmax 返回 0 会命中
    # 零概率列）：回退到最后一个非零列，与纯 Python 版契约一致
    bad = ~mask.any(dim=-1)
    if bool(bad.any()):
        nz = probs > 0
        last_nonzero = (nz.long().cumsum(dim=-1) * nz).argmax(dim=-1)
        out = torch.where(bad, last_nonzero, out)
    return out.int()


# torch 仅在 random_sampling_ref 内部局部导入：CPU 调度栈不依赖 torch。


__all__ = [
    "CASE_SAMPLE",
    "CASE_NEXTN",
    "CASE_VERIFY",
    "splitmix64_to_uniform",
    "derive_uniform",
    "derive_uniform_nextn",
    "sample_cdf_inverse",
    "random_sampling_ref_pure",
    "random_sampling_ref",
]

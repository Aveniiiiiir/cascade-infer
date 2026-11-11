"""det_random_sampling：确定性采样的 Triton kernel（bit 级对齐参考实现）。

随机流（DeepInfer random_sampling.py:70-84 的逐位复刻）：
    seq      = ((pos * 3 + case) * GOLDEN) mod 2^64
    combined = (seed + (seq ^ (seq << 13))) mod 2^64
    u        = SplitMix64(combined) → fp64 [0, 1)
    out      = 首个满足 CDF > u·total 的列（逆 CDF；零概率列永不命中）

Triton 实现要点：
- uint64 语义用 int64 承载（乘法/加法按二补码回绕，位模式一致）；
- 逻辑右移用 `(x >> n) & ((1 << (64-n)) - 1)` 模拟（算术移位补 1 后掩掉）；
- `seq << 13` 用 `seq * 8192`（回绕等价）；
- SplitMix64 常数以 int64 标量参数传入（避免字面量符号问题）；
- 逆 CDF 两遍扫描（total → 穿越 + 最后非零列回退），`tl.cumsum` fp64；
- 每行独立计算 ⇒ 行顺序无关（批不变由构造保证）。
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.common import CASE_SAMPLE
from cascade.model_executor.triton_ops.ops import tl, triton, use_triton
from cascade.sampling.rng import _i64, random_sampling_ref, random_sampling_ref_pure
from cascade.sampling_params import MTP_SEED_OFFSET

_GOLDEN_I64 = _i64(0x9E3779B97F4A7C15)
_SM_A_I64 = _i64(0xBF58476D1CE4E5B9)
_SM_B_I64 = _i64(0x94D049BB133111EB)
_2_POW_NEG_53 = 2.0**-53

_BLOCK_V = 1024
_BIG = 2147483647  # int32 哨兵（vocab < 2^31）


if triton is not None:

    @triton.jit
    def _det_random_sampling_kernel(
        probs_ptr,  # [T, vocab] fp32 连续（行和 ≤ 1，零概率列允许）
        seeds_ptr,  # [T] int64（uint64 位模式；draft 流偏移已在 wrapper 完成）
        positions_ptr,  # [T] int64 绝对位置
        out_ptr,  # [T] int32
        vocab,
        GOLDEN,  # int64 标量：0x9E3779B97F4A7C15 的二补码
        SM_A,  # int64 标量
        SM_B,  # int64 标量
        CASE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        t = tl.program_id(0)
        seed = tl.load(seeds_ptr + t)
        pos = tl.load(positions_ptr + t)
        # combined = seed + (seq ^ (seq << 13))，seq = (pos*3 + case) * GOLDEN
        seq = (pos * 3 + CASE) * GOLDEN
        mixed = seq ^ (seq * 8192)  # x<<13 ≡ x*8192 (mod 2^64)
        combined = seed + mixed
        # SplitMix64 finalizer（逻辑右移 = 算术移位 + 掩码）
        z = combined + GOLDEN
        z = (z ^ ((z >> 30) & 17179869183)) * SM_A  # (1<<34)-1
        z = (z ^ ((z >> 27) & 137438953471)) * SM_B  # (1<<37)-1
        z = z ^ ((z >> 31) & 8589934591)  # (1<<33)-1
        u = ((z >> 11) & 9007199254740991).to(tl.float64) * 1.1102230246251565e-16  # (1<<53)-1, 2^-53
        # 第一遍：行总概率
        total = 0.0
        for v0 in range(0, vocab, BLOCK_V):
            v = v0 + tl.arange(0, BLOCK_V)
            p = tl.load(probs_ptr + t * vocab + v, mask=v < vocab, other=0.0).to(tl.float64)
            total += tl.sum(p, axis=0)
        target = u * total
        # 第二遍：首个穿越列 + 最后非零列（次正规数边界的回退契约）
        cdf = 0.0
        best = _BIG
        last_nz = 0
        for v0 in range(0, vocab, BLOCK_V):
            v = v0 + tl.arange(0, BLOCK_V)
            p = tl.load(probs_ptr + t * vocab + v, mask=v < vocab, other=0.0).to(tl.float64)
            c = tl.cumsum(p, axis=0) + cdf
            cross = c > target
            cand = tl.where(cross, v, _BIG)
            best = tl.minimum(best, tl.min(cand, axis=0))
            nz = p > 0
            nz_idx = tl.where(nz, v, -1)
            last_nz = tl.maximum(last_nz, tl.max(nz_idx, axis=0))
            cdf += tl.sum(p, axis=0)
        out_idx = tl.where(best < _BIG, best, last_nz)
        tl.store(out_ptr + t, out_idx.to(tl.int32))


def det_random_sampling(
    probs: torch.Tensor,  # [T, vocab] fp32 连续
    seq_seeds: torch.Tensor,  # [T]（uint64/int64；draft 流偏移由 seed_offset 施加）
    sample_positions: torch.Tensor,  # [T] 绝对 token 位置
    case: int = CASE_SAMPLE,
    seed_offset: int = 0,
) -> torch.Tensor:
    """确定性采样 → [T] int32。case: 0=普通 1=nextn 2=verify。"""
    assert probs.ndim == 2 and probs.is_contiguous() and probs.dtype == torch.float32
    T, vocab = probs.shape
    if T == 0:
        return torch.empty(0, dtype=torch.int32, device=probs.device)
    seeds = seq_seeds.to(torch.int64)
    if seed_offset:
        seeds = seeds + _i64(seed_offset % (1 << 64))
    positions = sample_positions.to(torch.int64)
    out = torch.empty(T, dtype=torch.int32, device=probs.device)
    if use_triton(probs.device):
        _det_random_sampling_kernel[(T,)](
            probs, seeds, positions, out, vocab,
            _GOLDEN_I64, _SM_A_I64, _SM_B_I64,
            CASE=case, BLOCK_V=_BLOCK_V, num_warps=8,
        )
        return out
    # ref 路径：无 triton 环境。torch 向量化版只覆盖两个实际使用的组合，
    # 其余（case=2 verify 流等）退回纯公式（慢但 bit 级一致）。
    if case == 0 and seed_offset == 0:
        return random_sampling_ref(probs, seq_seeds, sample_positions)
    if case == 1 and seed_offset == MTP_SEED_OFFSET:
        return random_sampling_ref(probs, seq_seeds, sample_positions, is_nextn_predict=True)
    pure = random_sampling_ref_pure(
        [r.tolist() for r in probs],
        [(s + seed_offset) % (1 << 64) for s in seq_seeds.tolist()],
        positions.tolist(),
        case=case,
    )
    return torch.tensor(pure, dtype=torch.int32, device=probs.device)


def det_random_sampling_nextn(
    probs: torch.Tensor, seq_seeds: torch.Tensor, sample_positions: torch.Tensor
) -> torch.Tensor:
    """draft/nextn 流（case=1 + MTP_SEED_OFFSET 独立偏移）。"""
    return det_random_sampling(
        probs, seq_seeds, sample_positions, case=1, seed_offset=MTP_SEED_OFFSET
    )


__all__ = ["det_random_sampling", "det_random_sampling_nextn"]

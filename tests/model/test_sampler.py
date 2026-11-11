"""Sampler 编排测试：变换 oracle 对拍、贪心退化、top-p 性质、golden、SO 掩码。"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from cascade.sampling.rng import derive_uniform, sample_cdf_inverse
from cascade.sampling.sampler import Sampler, SamplingRows, build_sampling_rows
from cascade.sampling_params import SamplingParams


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rows(device, scalars_list, positions, goldens=None):
    per_row = [
        (f"r{i}", s, positions[i], (goldens[i] if goldens else -1))
        for i, s in enumerate(scalars_list)
    ]
    return build_sampling_rows(per_row, device)


def _naive_transform_row(logits_row: torch.Tensor, temp: float, k: int, p: float) -> list[float]:
    """naive oracle：单行 温度 → top-k → top-p → softmax。"""
    x = (logits_row / temp).tolist()
    if k > 0 and k < len(x):
        kth = sorted(x, reverse=True)[k - 1]
        x = [(-math.inf if v < kth else v) for v in x]
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    total = sum(exps)
    probs = [e / total for e in exps]
    if p < 1.0:
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        cum = 0.0
        keep = [False] * len(probs)
        for i in order:
            if cum < p:
                keep[i] = True
                cum += probs[i]
            else:
                break
        probs = [pr if keep[i] else 0.0 for i, pr in enumerate(probs)]
    return probs


def _naive_sample(logits_row, temp, k, p, seed, pos) -> int:
    probs = _naive_transform_row(logits_row, temp, k, p)
    u = derive_uniform(seed, pos, 0)
    return sample_cdf_inverse(probs, u)


@pytest.mark.parametrize("device", _devices())
def test_matches_naive_oracle(device: str) -> None:
    torch.manual_seed(0)
    R, vocab = 48, 512
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    params = [
        SamplingParams(temperature=0.8, top_p=0.9, top_k=50, seq_seed=1000 + i, max_tokens=10)
        for i in range(R)
    ]
    positions = [37 * i for i in range(R)]
    rows = _rows(device, [p.snapshot() for p in params], positions)
    sampler = Sampler()
    got = sampler.sample(logits.clone(), rows)
    for i in range(R):
        exp = _naive_sample(
            logits[i].cpu(), params[i].temperature, params[i].top_k, params[i].top_p,
            params[i].seq_seed, positions[i],
        )
        assert got[i] == exp, f"row {i}: {got[i]} != {exp}"


@pytest.mark.parametrize("device", _devices())
def test_greedy_equals_argmax(device: str) -> None:
    """温度 < 1e-5 的退化路径 = argmax（tie 取最小 index）。"""
    torch.manual_seed(1)
    R, vocab = 64, 1024
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    # 注入并列最大值，验证 tie 语义
    logits[3, 5] = logits[3, 100] = logits[3].max() + 1.0
    params = [SamplingParams(temperature=0.0, seq_seed=i, max_tokens=10) for i in range(R)]
    rows = _rows(device, [p.snapshot() for p in params], [i for i in range(R)])
    got = Sampler().sample(logits.clone(), rows)
    argmax = logits.argmax(dim=-1).tolist()
    assert got == argmax


@pytest.mark.parametrize("device", _devices())
def test_top_k_one_equals_greedy(device: str) -> None:
    torch.manual_seed(2)
    R, vocab = 32, 256
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    params = [SamplingParams(temperature=1.0, top_k=1, seq_seed=i, max_tokens=10) for i in range(R)]
    rows = _rows(device, [p.snapshot() for p in params], [i for i in range(R)])
    got = Sampler().sample(logits.clone(), rows)
    assert got == logits.argmax(dim=-1).tolist()


@pytest.mark.parametrize("device", _devices())
def test_top_p_prefix_property(device: str) -> None:
    """top-p 采样必落在保留前缀内（累积 < p 的 token ∪ 首个 token）。"""
    torch.manual_seed(3)
    R, vocab = 64, 128
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    p = 0.6
    params = [SamplingParams(temperature=1.0, top_p=p, seq_seed=i, max_tokens=10) for i in range(R)]
    rows = _rows(device, [p_.snapshot() for p_ in params], [i for i in range(R)])
    got = Sampler().sample(logits.clone(), rows)
    probs = torch.softmax(logits, dim=-1)
    for i in range(R):
        sp, _ = probs[i].sort(descending=True)
        cum = sp.cumsum(0)
        keep_n = int((cum - sp < p).sum().item()) + 1  # 判据 + 首个
        threshold = sp[keep_n - 1].item()
        assert probs[i, got[i]] >= threshold, (
            f"row {i}: 采到 top-p 前缀外 token（p={probs[i, got[i]].item()} < {threshold}）"
        )


@pytest.mark.parametrize("device", _devices())
def test_golden_teacher_forcing(device: str) -> None:
    torch.manual_seed(4)
    R, vocab = 8, 64
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    params = [SamplingParams(temperature=1.0, seq_seed=i, max_tokens=10) for i in range(R)]
    goldens = [3, 7, 11, 19, 23, 31, 41, 53]
    # golden_token_ids 的第 output_len 个生效：snapshot 已解析为当前值
    snaps = [replace(p.snapshot(), golden_token_ids=(goldens[i],)) for i, p in enumerate(params)]
    rows = _rows(device, snaps, [i for i in range(R)], goldens)
    got = Sampler().sample(logits.clone(), rows)
    assert got == goldens


@pytest.mark.parametrize("device", _devices())
def test_so_bitmask_constraint(device: str) -> None:
    """SO 掩码在变换前生效：采样必落在合法 token 集内。"""
    torch.manual_seed(5)
    R, vocab = 16, 256
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    num_words = (vocab + 31) // 32
    g = torch.Generator().manual_seed(6)
    words = torch.randint(0, 2**31, (R, num_words), generator=g, dtype=torch.int64).to(torch.int32).to(device)
    # 每行至少 8 个合法位：直接点亮 token 0..7 对应的位
    for i in range(R):
        allowed = ((words[i].long() >> (torch.arange(vocab, device=device) % 32)) & 1).bool()
        if int(allowed.sum()) < 8:
            for j in range(8):
                words[i, j // 32] |= 1 << (j % 32)
    params = [SamplingParams(temperature=1.0, seq_seed=i, max_tokens=10) for i in range(R)]
    rows = _rows(device, [p.snapshot() for p in params], [i for i in range(R)])
    row_map = torch.arange(R, dtype=torch.int32, device=device)
    got = Sampler().sample(logits.clone(), rows, so_bitmask=words, so_row_map=row_map)
    idx = torch.arange(vocab, device=device)
    w = idx // 32
    b = idx % 32
    for i in range(R):
        allowed = ((words[i].long()[w] >> b) & 1).bool()
        assert allowed[got[i]], f"row {i}: 采到被 SO 禁止的 token {got[i]}"


@pytest.mark.parametrize("device", _devices())
def test_determinism_and_row_order(device: str) -> None:
    torch.manual_seed(7)
    R, vocab = 96, 512
    logits = torch.randn(R, vocab, dtype=torch.float32, device=device)
    params = [SamplingParams(temperature=0.7, top_p=0.95, seq_seed=i, max_tokens=10) for i in range(R)]
    rows = _rows(device, [p.snapshot() for p in params], [17 * i for i in range(R)])
    a = Sampler().sample(logits.clone(), rows)
    b = Sampler().sample(logits.clone(), rows)
    assert a == b
    g = torch.Generator().manual_seed(8)
    perm = torch.randperm(R, generator=g).to(device)
    rows_p = _rows(device, [params[i].snapshot() for i in perm.tolist()], [17 * i for i in perm.tolist()])
    c = Sampler().sample(logits[perm].contiguous(), rows_p)
    assert c == [a[i] for i in perm.tolist()]


@pytest.mark.parametrize("device", _devices())
def test_sampling_distribution_goodness_of_fit(device: str) -> None:
    """卡方 GoF：大样本经验频率 ≈ 变换后分布（KS 检验的等价验收）。"""
    torch.manual_seed(9)
    vocab = 64
    T = 20_000
    logits_row = torch.randn(vocab, dtype=torch.float32)
    logits = logits_row.repeat(T, 1).to(device)
    params = [
        SamplingParams(temperature=1.0, top_p=0.95, top_k=-1, seq_seed=i, max_tokens=10)
        for i in range(1)  # snapshot 共享（seed 逐行不同）
    ]
    per_row = [
        (f"r{i}", params[0].snapshot(), i, -1) for i in range(T)
    ]
    rows = build_sampling_rows(per_row, device)
    got = Sampler().sample(logits.clone(), rows)
    counts = torch.bincount(torch.tensor(got, dtype=torch.long), minlength=vocab).float()
    # 期望分布 = naive 变换后的 probs（独立 oracle）
    probs = torch.softmax(logits_row / 1.0, dim=-1)
    exp_probs = _naive_transform_row(logits_row, 1.0, -1, 0.95)
    expected = torch.tensor(exp_probs, device=device) * T
    chi2 = ((counts - expected) ** 2 / expected.clamp_min(1e-6)).sum().item()
    # 自由度 ≈ vocab-1 = 63：均值 63，σ = sqrt(2*63) ≈ 11.2；5σ 上界
    assert chi2 < 63 + 5 * (2 * 63) ** 0.5, f"卡方 {chi2:.1f} 超界（分布偏移）"

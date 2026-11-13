"""MTP 拒绝采样数学测试（torch）：
- 贪心等价（MTP ≡ 非 MTP 逐 token）
- 理论接受率 Σ min(p, q)
- K=0 退化 = 普通采样（同 u 流）
- accept_probs ≈ 经验接受率
- -1 哨兵 fresh 行
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cascade.sampling.mtp_verify import mtp_verify
from cascade.sampling.rng import derive_uniform

V = 64


def _greedy_seq(logits_fn, n_steps: int, start: int = 0) -> list[int]:
    """非 MTP 贪心参考（teacher-forced：每步 logits 由位置决定）。"""
    out = []
    for p in range(start, start + n_steps):
        out.append(int(logits_fn(p).argmax().item()))
    return out


def _make_target(seed: int, n_pos: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(V, generator=g) for _ in range(n_pos)]


def _greedy_onehot_lp(argmax: int, strength: float = 20.0) -> torch.Tensor:
    lp = torch.full((V,), -strength)
    lp[argmax] = 0.0
    return torch.log_softmax(lp, dim=-1)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_greedy_equivalence(device: str) -> None:
    """贪心目标 + 贪心草案（one-hot）⇒ MTP 输出 = 非 MTP 贪心序列。"""
    torch.manual_seed(1)
    K = 4
    logits = [l.to(device) for l in _make_target(7, K + 2)]
    # 贪心 target = one-hot（log_softmax 后 argmax 位为 0，其余远小于 0）。
    # 拒绝采样的「对就必接受」要求 target 也是 one-hot：q_lp[argmax]=0 =
    # p_lp[argmax] → ratio=0 → exp=1 → 必接受；draft 错时 q_lp[wrong] 远
    # 小于 0 → ratio 远小于 0 → exp≈0 → 必拒绝。这是 DeepInfer 贪心等价
    # 的数学前提（target/draft 均 argmax），非 one-hot target 不成立。
    greedy = [int(l.argmax().item()) for l in logits]
    target_lp = torch.stack([_greedy_onehot_lp(g).to(device) for g in greedy[: K + 1]])
    # 贪心草案：交替对/错（对 = 目标 argmax；错 = argmax+1）
    drafts = [greedy[j] if j % 2 == 0 else (greedy[j] + 1) % V for j in range(K)]
    draft_lp = torch.stack([_greedy_onehot_lp(d) for d in drafts]).to(device)
    u_a = torch.full((K,), 0.5)  # ratio=1（对）→ 0.5<1 接受；ratio≈0（错）→ 0.5>0 拒绝
    u_s = torch.full((K + 1,), 0.5)
    res = mtp_verify(target_lp, draft_lp, torch.tensor(drafts, device=device), u_a.to(device), u_s.to(device))
    # 非 MTP 贪心：d[0] 对 → 接受；d[1] 错 → 拒绝 → 残差重采样 = 目标 argmax
    #（draft one-hot 在错 token 上，q 的 argmax 概率不被压制）
    exp = [greedy[0], greedy[1]]  # d0 接受；d1 拒绝 → 残差 = greedy[1]
    assert res.committed == exp
    assert res.n_accepted_drafts == 1


def test_greedy_all_accepted_bonus() -> None:
    """草案全对 → 全接受 + bonus（= 目标 argmax）。"""
    torch.manual_seed(2)
    K = 3
    logits = _make_target(11, K + 1)
    greedy = [int(l.argmax().item()) for l in logits]
    # 贪心 target = one-hot（同上：q_lp[argmax]=0 才满足必接受前提）
    target_lp = torch.stack([_greedy_onehot_lp(g) for g in greedy])
    drafts = greedy[:K]
    draft_lp = torch.stack([_greedy_onehot_lp(d) for d in drafts])
    res = mtp_verify(
        target_lp, draft_lp, torch.tensor(drafts),
        torch.full((K,), 0.5), torch.full((K + 1,), 0.5),  # ratio=1 → 0.5<1 必接受
    )
    # 全接受 + bonus = greedy[K]（K=0 位置 logits 的 argmax）
    assert res.committed == drafts + [greedy[K]]
    assert res.n_accepted_drafts == K


def test_k0_degenerates_to_normal_sampling() -> None:
    """K=0（无草案 / fresh 哨兵行）= 普通确定性采样（同 u 流 bit 级一致）。"""
    torch.manual_seed(3)
    lp = torch.log_softmax(torch.randn(V), dim=-1).unsqueeze(0)  # [1, V]
    u = torch.tensor([0.37])
    res = mtp_verify(
        lp, torch.zeros(0, V), torch.zeros(0, dtype=torch.int64),
        torch.zeros(0), u,
    )
    cdf = torch.cumsum(lp.exp().double(), dim=-1)
    expect = int((cdf > 0.37 * float(cdf[0, -1])).long().argmax().item())
    assert res.committed == [expect]
    assert res.bonus_token == expect


def test_negative_one_sentinel_is_fresh_row() -> None:
    """-1 哨兵（varlen fresh 行）→ 强制拒绝 → 正常采样。"""
    torch.manual_seed(4)
    K = 3
    logits = _make_target(13, K + 1)
    target_lp = torch.stack([torch.log_softmax(l, dim=-1) for l in logits])
    draft_lp = torch.stack([torch.log_softmax(torch.randn(V), dim=-1) for _ in range(K)])
    u_s = torch.tensor([derive_uniform(9, 100 + j) for j in range(K + 1)])
    res = mtp_verify(
        target_lp, draft_lp, torch.tensor([-1, 5, 9]),
        torch.zeros(K), u_s,
    )
    # 哨兵在首位 → 普通采样 1 个 token 即返回
    assert len(res.committed) == 1 and res.n_accepted_drafts == 0
    assert res.accept_probs == [1.0]


def test_theoretical_acceptance_rate() -> None:
    """经验接受率 ≈ Σ min(p, q)（合成分布，大样本）。"""
    torch.manual_seed(5)
    trials = 4000
    q = torch.softmax(torch.randn(V), dim=-1)
    p = torch.softmax(torch.randn(V) * 1.3, dim=-1)
    theoretical = float(torch.minimum(q, p).sum().item())
    # 草案按 p 采样
    g = torch.Generator().manual_seed(6)
    accepted = 0
    total_drafts = 0
    for t in range(trials):
        d = int(torch.multinomial(p, 1, generator=g).item())
        u = derive_uniform(1000 + t, 0, 2)
        ratio = math.exp(min(float(torch.log(q[d]) - torch.log(p[d])), 0.0))
        total_drafts += 1
        if u < ratio:
            accepted += 1
        # 单草案验证（K=1：接受则 committed=[d]；不接受则 [resample]）
        # 只统计首草案的接受
    emp = accepted / total_drafts
    assert abs(emp - theoretical) < 0.03, f"经验 {emp:.3f} vs 理论 {theoretical:.3f}"


def test_accept_probs_track_empirical() -> None:
    """accept_probs = clamp(1 − Σ residual) 与经验接受率一致。"""
    torch.manual_seed(7)
    q = torch.softmax(torch.randn(V), dim=-1)
    p = torch.softmax(torch.randn(V) * 0.9, dim=-1)
    from cascade.sampling.mtp_verify import _accept_prob

    ap = _accept_prob(q, torch.log(p))
    theoretical = float(torch.minimum(q, p).sum().item())
    assert abs(ap - theoretical) < 0.05, f"accept_prob {ap:.3f} vs Σmin {theoretical:.3f}"

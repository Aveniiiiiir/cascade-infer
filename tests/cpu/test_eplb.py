"""EPLB 策略测试（CPU，本机可跑）：结构自洽 + 确定性 + 负载改善。"""

from __future__ import annotations

from cascade.distributed.eplb import (
    derive_num_physical,
    eplb_load_balance_ratio,
    rebalance_experts_global,
)


def test_derive_num_physical() -> None:
    assert derive_num_physical(64, 8) == 64  # 无冗余
    assert derive_num_physical(64, 8, min_extra=4) == 72  # 64+4 → 72（8 的倍数）
    assert derive_num_physical(60, 8) == 64  # 向上取整到倍数
    assert derive_num_physical(100, 3) == 102


def test_rebalance_structure_invariants() -> None:
    E, W, P = 16, 4, 24  # 8 个冗余副本
    counts = [10 + i * 5 for i in range(E)]  # 递增负载
    l2p, p2l = rebalance_experts_global(counts, E, W, P)
    # 1. 每专家至少一个 physical 槽
    assert all(len(v) >= 1 for v in l2p)
    # 2. physical 恰好占满且映射一致
    assert sum(len(v) for v in l2p) == P
    for e, plist in enumerate(l2p):
        for pid in plist:
            assert p2l[pid] == e
    assert all(x >= 0 for x in p2l)
    # 3. 每 rank 容量 = P/W
    capacity = P // W
    per_rank = [0] * W
    for pid, e in enumerate(p2l):
        per_rank[pid // capacity] += 1
    assert all(c == capacity for c in per_rank)
    # 4. 热点专家拿到更多副本
    hottest = counts.index(max(counts))
    coldest = counts.index(min(counts))
    assert len(l2p[hottest]) >= len(l2p[coldest])
    assert len(l2p[hottest]) > 1  # 有冗余才会被复制


def test_rebalance_deterministic() -> None:
    E, W, P = 8, 2, 12
    counts = [3, 100, 5, 7, 2, 50, 1, 9]
    a = rebalance_experts_global(counts, E, W, P)
    b = rebalance_experts_global(counts, E, W, P)
    assert a == b


def test_rebalance_improves_balance() -> None:
    """重排后按 placement 计算的 moe_lb 优于按顺序放置的基线。"""
    E, W, P = 16, 4, 16  # 无冗余：纯放置
    counts = [0] * E
    counts[0] = 1000  # 单一热点
    counts[1] = 900
    counts[2] = 800
    l2p, p2l = rebalance_experts_global(counts, E, W, P)
    # 重排后每 rank 的负载
    capacity = P // W
    per_rank = [0] * W
    for pid, e in enumerate(p2l):
        per_rank[pid // capacity] += counts[e]
    balanced = max(per_rank) / (sum(per_rank) / W)
    # 顺序放置的基线：expert i → rank i // (E/W)
    base_rank = [0] * W
    per = E // W
    for e, c in enumerate(counts):
        base_rank[e // per] += c
    baseline = max(base_rank) / (sum(base_rank) / W)
    assert balanced < baseline, f"重排未改善: {balanced:.2f} vs {baseline:.2f}"


def test_moe_lb_ratio() -> None:
    counts = [10, 10, 10, 10]
    assert eplb_load_balance_ratio(counts, 4, 2) == 1.0
    counts = [40, 0, 0, 0]
    assert eplb_load_balance_ratio(counts, 4, 2) == 2.0  # rank0=40, rank1=0 → 40/(40/2)

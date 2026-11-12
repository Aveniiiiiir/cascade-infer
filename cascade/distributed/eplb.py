"""EPLB（专家并行负载均衡）策略：冗余分配 + 贪心放置。

DeepInfer eplb/policy.py 语义（结构自洽 + 确定性；exact 打分公式以其
源码为准，本实现为等价目标的贪心）：
- **冗余分配**：physical = logical + n_extra（整除 ep_size）。每轮复制
  一个"当前每副本峰值负载最高的"专家（复制热点 ⇒ 最小化最大每副本负载）
- **放置**：副本按负载降序逐个放入"累积负载最小且有余量的 rank"
  （容量 = physical / ep_size），多步历史下取各步峰值负载最小的 rank
- 产出：logical_to_physical（含副本列表）/ physical_to_logical /
  logical_count（每专家副本数）——权重搬运由上层（NCCL P2P）执行

批不变注记：重排期间 DeepInfer 强制 EP 单次归约（report A 第 4 条）；
本模块纯 CPU 策略不影响归约序。
"""

from __future__ import annotations


def derive_num_physical(num_experts: int, ep_size: int, min_extra: int = 0) -> int:
    """physical = logical + extra，向上取整到 ep_size 的倍数（DeepInfer
    OVERRIDE 默认下限同样语义）。"""
    assert num_experts > 0 and ep_size > 0
    extra = max(min_extra, 0)
    total = num_experts + extra
    # 取 ≥ total 的最小 ep_size 倍数（至少 1 倍）
    multiple = max(1, (total + ep_size - 1) // ep_size)
    return multiple * ep_size


def rebalance_experts_global(
    counts: list[list[int]],  # [E]：窗口内每专家的 token 计数（单层语义）
    num_experts: int,
    ep_size: int,
    num_physical: int,
) -> tuple[list[list[int]], list[int]]:
    """单层重排 → (logical_to_physical [E][copies], physical_to_logical [P])。

    多层模型逐层独立调用（DeepInfer 同构布局分组多 balancer）。
    确定性：纯函数，无随机。
    """
    E, P = num_experts, num_physical
    assert len(counts) == E and P >= E and P % ep_size == 0
    extra = P - E
    # ---- 1. 冗余分配：复制热点（每副本负载 = count / copies）----
    copies = [1] * E
    for _ in range(extra):
        best_e, best_load = -1, -1.0
        for e in range(E):
            per_copy = counts[e] / copies[e]  # 副本均摊后的每副本负载
            if per_copy > best_load:
                best_load, best_e = per_copy, e
        copies[best_e] += 1
    # ---- 2. 放置：负载降序 → 最小累积负载且有余量的 rank ----
    capacity = P // ep_size
    rank_load = [0] * ep_size  # 累积负载
    rank_used = [0] * ep_size  # 已占槽数
    slots: list[tuple[int, int]] = []  # (rank, slot_idx) per physical id 顺序
    physical_to_logical: list[int] = [-1] * P
    # 副本按负载降序（count/copies；确定性 tie-break 按 expert id）
    order = sorted(
        ((e, c) for e in range(E) for c in range(copies[e])),
        key=lambda ec: (-(counts[ec[0]] / copies[ec[0]]), ec[0], ec[1]),
    )
    # 逐 rank 槽位（贪心放最小负载 rank）
    placements: dict[tuple[int, int], int] = {}  # (e, copy) → physical id
    for e, c in order:
        # 选 rank：容量有余且负载最小（tie → rank id 小）
        best_r, best_l = -1, None
        for r in range(ep_size):
            if rank_used[r] < capacity and (best_l is None or rank_load[r] < best_l):
                best_l, best_r = rank_load[r], r
        assert best_r >= 0, "容量不足（num_physical 不是 ep_size 倍数？）"
        physical_id = best_r * capacity + rank_used[best_r]
        rank_used[best_r] += 1
        rank_load[best_r] += counts[e] / copies[e]
        placements[(e, c)] = physical_id
        physical_to_logical[physical_id] = e
        slots.append((best_r, physical_id))
    # logical → physical 列表
    logical_to_physical: list[list[int]] = [[] for _ in range(E)]
    for (e, _c), pid in placements.items():
        logical_to_physical[e].append(pid)
    for e in range(E):
        assert logical_to_physical[e], f"expert {e} 未放置"
        logical_to_physical[e].sort()
    return logical_to_physical, physical_to_logical


def eplb_load_balance_ratio(counts: list[int], num_experts: int, ep_size: int) -> float:
    """moe_lb 指标 = max_rank / mean（DeepInfer MOE_LB_STAT_INTERVAL 语义）。"""
    assert len(counts) == num_experts
    per_rank = [0] * ep_size
    cap = num_experts // ep_size
    for e, c in enumerate(counts):
        per_rank[e // cap] += c
    total = sum(per_rank)
    if total == 0:
        return 1.0
    return max(per_rank) / (total / ep_size)


__all__ = [
    "derive_num_physical",
    "rebalance_experts_global",
    "eplb_load_balance_ratio",
]

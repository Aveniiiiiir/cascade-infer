"""BlockManager unit + fuzz tests (Phase 1 acceptance gate)."""

from __future__ import annotations

import random

import pytest

from cascade.core.block_manager import BlockAllocationError, BlockManager


def test_allocate_free_cycle() -> None:
    bm = BlockManager(8)
    assert bm.num_free_blocks == 8
    b = bm.allocate()
    assert 1 <= b <= 8
    assert bm.num_free_blocks == 7
    bm.release(b)
    assert bm.num_free_blocks == 8
    bm.debug_check_consistency()


def test_block_zero_reserved() -> None:
    bm = BlockManager(4)
    seen = {bm.allocate() for _ in range(4)}
    assert 0 not in seen
    with pytest.raises(BlockAllocationError):
        bm.allocate()


def test_ref_pin_unpin() -> None:
    bm = BlockManager(4)
    b = bm.allocate()
    bm.retain(b)  # fork: 另一个 seq 共享
    assert bm.ref_count(b) == 2
    bm.release(b)
    assert bm.ref_count(b) == 1
    assert bm.num_free_blocks == 3  # 还有一个持有者，未回收
    bm.release(b)
    assert bm.num_free_blocks == 4
    bm.debug_check_consistency()


def test_release_overcount_raises() -> None:
    bm = BlockManager(2)
    b = bm.allocate()
    bm.release(b)
    with pytest.raises(ValueError):
        bm.release(b)


def test_cache_ownership_keeps_ref0_blocks() -> None:
    bm = BlockManager(2)
    b = bm.allocate()
    bm.mark_cached(b)
    bm.release(b)  # ref 归零但 cache 持有 → 不回 free 池
    assert bm.num_free_blocks == 1
    assert bm.num_evictable_blocks == 1
    bm.unmark_cached(b)  # 逐出 → 回收
    assert bm.num_free_blocks == 2
    assert bm.num_evictable_blocks == 0
    bm.debug_check_consistency()


def test_high_water() -> None:
    bm = BlockManager(16)
    blocks = [bm.allocate() for _ in range(10)]
    assert bm.high_water_mark == 10
    for b in blocks[:4]:
        bm.release(b)
    assert bm.high_water_mark == 10


def test_fuzz_shadow_model() -> None:
    """随机 alloc/release/retain/mark/unmark 序列与影子模型逐步严格比对。

    影子状态约定:
    - shadow_ref 只含 ref>0 的块（ref 归零即移除，无论是否 cache 持有）
    - shadow_cached 为 cache 持有集（其中可能含 ref==0 的可逐出块）
    不变式: allocated == |shadow_ref ∪ shadow_cached|,
    free == num_blocks - allocated, ref_count 与影子一致。
    """
    rng = random.Random(0xC0FFEE)
    bm = BlockManager(64)
    shadow_ref: dict[int, int] = {}
    shadow_cached: set[int] = set()
    for _ in range(20_000):
        ops: list[str] = ["alloc"]
        if shadow_ref:
            ops += ["release", "retain"]
        # ref==0 的 cache 块可被重新 pin（前缀命中语义）
        evictable = shadow_cached - set(shadow_ref)
        if evictable:
            ops.append("pin_cached")
        if shadow_ref:
            ops.append("mark")
        if shadow_cached:
            ops.append("unmark")
        op = rng.choice(ops)
        if op == "alloc":
            if bm.num_free_blocks == 0:
                continue
            b = bm.allocate()
            assert b not in shadow_ref and b not in shadow_cached
            shadow_ref[b] = 1
        elif op == "release":
            b = rng.choice(list(shadow_ref))
            bm.release(b)
            shadow_ref[b] -= 1
            if shadow_ref[b] == 0:
                del shadow_ref[b]  # free 或 cache 持有（后者在 shadow_cached 中）
        elif op == "retain":
            b = rng.choice(list(shadow_ref))
            bm.retain(b)
            shadow_ref[b] += 1
        elif op == "pin_cached":
            b = rng.choice(sorted(evictable))
            bm.retain(b)  # match 命中：ref 0→1
            shadow_ref[b] = 1
        elif op == "mark":
            eligible = [b for b in shadow_ref if b not in shadow_cached]
            if eligible:
                b = rng.choice(eligible)
                bm.mark_cached(b)
                shadow_cached.add(b)
        elif op == "unmark":
            b = rng.choice(sorted(shadow_cached))
            bm.unmark_cached(b)
            shadow_cached.discard(b)
            if shadow_ref.get(b) == 0:
                shadow_ref.pop(b, None)
        # ---- 逐步严格比对 ----
        allocated = len(set(shadow_ref) | shadow_cached)
        assert bm.num_allocated_blocks == allocated, (
            f"allocated {bm.num_allocated_blocks} != shadow {allocated} "
            f"(ref={shadow_ref} cached={sorted(shadow_cached)})"
        )
        assert bm.num_free_blocks == bm.num_blocks - allocated
        for b in set(shadow_ref) | shadow_cached:
            assert bm.ref_count(b) == shadow_ref.get(b, 0)
        bm.debug_check_consistency()


def test_fuzz_no_leak_eventually_all_free() -> None:
    """fuzz 结束后全部释放 + 逐出 → 池满复原（无泄漏）。"""
    rng = random.Random(42)
    bm = BlockManager(32)
    live: list[int] = []
    for _ in range(5_000):
        action = rng.random()
        if action < 0.55 and bm.num_free_blocks > 0:
            live.append(bm.allocate())
        elif live:
            i = rng.randrange(len(live))
            b = live.pop(i)
            while bm.ref_count(b) > 0:
                bm.release(b)
    # 清理 cache 持有
    for b in range(1, bm.num_blocks + 1):
        while bm.ref_count(b) > 0:
            bm.release(b)
        if bm.is_cache_owned(b):
            bm.unmark_cached(b)
    assert bm.num_free_blocks == bm.num_blocks
    bm.debug_check_consistency()

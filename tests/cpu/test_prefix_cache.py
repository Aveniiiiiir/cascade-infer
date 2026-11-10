"""PrefixCache unit tests: hash chain, LRU eviction, pin semantics.

块生命周期约定：allocate（ref=1）→ insert 注册（cache-owned）→ release
（seq 结束，ref=0，可逐出）→ match 命中（retain pin，ref=1）→ 请求结束
release → 可再次逐出。
"""

from __future__ import annotations

from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache

BS = 8  # 小块便于构造


def make_tokens(prefix: int, n: int, salt: int = 0) -> list[int]:
    return [(prefix * 131 + i * 7 + salt) % 9000 for i in range(n)]


def make_cache(num_blocks: int = 64) -> tuple[BlockManager, PrefixCache]:
    bm = BlockManager(num_blocks)
    return bm, PrefixCache(bm, BS)


def register(bm: BlockManager, pc: PrefixCache, tokens: list[int]) -> int:
    """模拟一个 seq 计算出一个满块并注册、随后结束（ref 归零可逐出）。"""
    b = bm.allocate()
    assert pc.insert_block(tokens, 0, b)
    bm.release(b)
    return b


def test_match_after_insert() -> None:
    bm, pc = make_cache()
    tokens = make_tokens(1, BS * 3)  # 3 个整块
    blocks = [bm.allocate() for _ in range(3)]
    pc.insert_full_blocks(tokens, blocks, 3)
    for b in blocks:  # seq 结束
        bm.release(b)
    hit, n = pc.match(tokens)
    assert len(hit) == 3 and n == BS * 3
    for b in hit:
        assert bm.ref_count(b) == 1  # match 会 pin
    pc.on_request_finished(hit)
    for b in blocks:
        assert bm.ref_count(b) == 0
    bm.debug_check_consistency()
    pc.debug_check_consistency()


def test_no_match_on_miss() -> None:
    bm, pc = make_cache()
    tokens = make_tokens(1, BS)
    register(bm, pc, tokens)
    hit, n = pc.match(make_tokens(2, BS))
    assert hit == [] and n == 0
    # 前缀更短（不足整块）也不命中
    hit, n = pc.match(tokens[: BS - 1])
    assert hit == [] and n == 0


def test_partial_prefix_match_stops_at_chain_break() -> None:
    bm, pc = make_cache()
    tokens_a = make_tokens(1, BS * 2)
    b1, b2 = bm.allocate(), bm.allocate()
    pc.insert_block(tokens_a, 0, b1)
    pc.insert_block(tokens_a, 1, b2)
    bm.release(b1)
    bm.release(b2)
    # 第二块内容不同 → 只命中第一块
    tokens_b = tokens_a[:BS] + make_tokens(7, BS)
    hit, n = pc.match(tokens_b)
    assert len(hit) == 1 and n == BS
    assert hit[0] == b1
    pc.on_request_finished(hit)
    pc.debug_check_consistency()


def test_chain_hash_requires_full_history() -> None:
    """链式哈希：同尾块、不同历史 → 不命中。"""
    bm, pc = make_cache()
    tokens_a = make_tokens(1, BS * 2)
    b1, b2 = bm.allocate(), bm.allocate()
    pc.insert_block(tokens_a, 0, b1)
    pc.insert_block(tokens_a, 1, b2)
    bm.release(b1)
    bm.release(b2)
    # 历史不同（第一块不同），尾块与 a 相同
    tokens_c = make_tokens(9, BS) + tokens_a[BS:]
    hit, n = pc.match(tokens_c)
    assert hit == [] and n == 0
    pc.debug_check_consistency()


def test_insert_dedup() -> None:
    bm, pc = make_cache()
    tokens = make_tokens(1, BS)
    b1, b2 = bm.allocate(), bm.allocate()
    assert pc.insert_block(tokens, 0, b1) is True
    assert pc.insert_block(tokens, 0, b2) is False  # 相同内容不重复注册
    assert bm.is_cache_owned(b1)
    assert not bm.is_cache_owned(b2)
    bm.release(b1)
    bm.release(b2)
    pc.debug_check_consistency()


def test_lru_eviction_order() -> None:
    bm, pc = make_cache(16)
    registered = [register(bm, pc, make_tokens(100 + i, BS)) for i in range(4)]
    evicted = pc.evict(2)
    assert evicted == 2
    assert not bm.is_cache_owned(registered[0])  # 最旧的先逐出
    assert not bm.is_cache_owned(registered[1])
    assert bm.is_cache_owned(registered[2])
    assert bm.is_cache_owned(registered[3])
    assert bm.num_free_blocks == 16 - 2  # 逐出的 2 块回收；其余 2 块 cache 持有
    pc.debug_check_consistency()


def test_pinned_blocks_not_evicted() -> None:
    bm, pc = make_cache()
    tokens = make_tokens(1, BS * 2)
    blocks = [bm.allocate() for _ in range(2)]
    pc.insert_full_blocks(tokens, blocks, 2)
    for b in blocks:  # owner 结束
        bm.release(b)
    hit, _ = pc.match(tokens)  # pin 全部
    assert pc.evict(4) == 0  # 全被 pin，逐不出
    assert bm.is_cache_owned(blocks[0])
    pc.on_request_finished(hit)
    assert pc.evict(1) == 1  # 释放后可逐出
    pc.debug_check_consistency()


def test_evict_respects_ref_semantics() -> None:
    """match 命中（ref>0）期间 evict 跳过该块；释放后才可逐。"""
    bm, pc = make_cache()
    b1 = register(bm, pc, make_tokens(1, BS))
    b2 = register(bm, pc, make_tokens(2, BS))
    # pin 住最旧的 b1
    hit, _ = pc.match(make_tokens(1, BS))
    assert pc.evict(1) == 1  # 只能逐 b2（跳过被 pin 的 b1）
    assert bm.is_cache_owned(b1) and not bm.is_cache_owned(b2)
    pc.on_request_finished(hit)
    assert pc.evict(1) == 1  # 现在能逐 b1
    assert bm.num_free_blocks == bm.num_blocks  # 全部回收
    bm.debug_check_consistency()
    pc.debug_check_consistency()


def test_hit_rate_stats() -> None:
    bm, pc = make_cache()
    register(bm, pc, make_tokens(1, BS))
    pc.match(make_tokens(1, BS))  # hit
    pc.match(make_tokens(3, BS))  # miss
    assert pc._queries == 2 and pc._hits == 1
    assert abs(pc.hit_rate - 0.5) < 1e-9

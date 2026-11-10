"""Prefix cache: block hash chain + LRU eviction + pin/unpin.

DeepInfer's open-source release ships no prefix cache (production uses an
external 3FS kvcache-store, see doc/inference_deployment_qa.md), so this module
follows the vLLM V1 design instead:

- A block's key is a *chained* hash: `H(prev_hash, block_tokens)` — a prefix
  hit requires the entire history to match, not just the last block.
- Only full blocks participate (a partial tail block cannot be a key).
- Hashing uses hashlib.blake2b (stdlib): stable across processes and platforms.
  Python's builtin hash() is process-randomized and must never be used here.
- Hits pin (retain) blocks; LRU eviction only touches ref==0 cache-owned
  blocks (see BlockManager ownership model).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from typing import Sequence

from cascade.common.logger import get_logger
from cascade.core.block_manager import BlockManager

_log = get_logger("core.prefix_cache")


class PrefixCache:
    __slots__ = ("block_size", "block_mgr", "_by_hash", "_lru", "_block_hash", "_hits", "_queries")

    def __init__(self, block_mgr: BlockManager, block_size: int) -> None:
        self.block_size = block_size
        self.block_mgr = block_mgr
        #: hash → 该内容哈希下的候选 block id 列表（多为单元素）
        self._by_hash: dict[int, deque[int]] = {}
        #: block_id → LRU 顺序（OrderedDict 值存 hash，便于淘汰时反查）
        self._lru: OrderedDict[int, int] = OrderedDict()
        #: block_id → hash（淘汰时从 _by_hash 移除用）
        self._block_hash: dict[int, int] = {}
        self._hits = 0
        self._queries = 0

    # ------------------------------------------------------------------ hashing

    def hash_block(self, prev_hash: int, token_ids: Sequence[int]) -> int:
        """链式哈希：H(prev_hash ‖ block tokens)。跨进程稳定（blake2b）。"""
        h = hashlib.blake2b(digest_size=8)
        h.update(prev_hash.to_bytes(8, "little"))
        for t in token_ids:
            h.update(int(t).to_bytes(8, "little"))
        return int.from_bytes(h.digest(), "little")

    # ------------------------------------------------------------------ match

    def match(self, token_ids: Sequence[int]) -> tuple[list[int], int]:
        """对 token 前缀做逐块哈希链匹配。

        Returns:
            (hit_block_ids, num_hit_tokens): 命中的 block 表（已 retain/pin，
            调用方负责在请求结束时 release）与命中的 token 数（block 对齐）。
            未命中时返回 ([], 0)。
        """
        self._queries += 1
        num_full_blocks = len(token_ids) // self.block_size
        hit_blocks: list[int] = []
        prev_hash = 0
        for i in range(num_full_blocks):
            chunk = token_ids[i * self.block_size : (i + 1) * self.block_size]
            h = self.hash_block(prev_hash, chunk)
            candidates = self._by_hash.get(h)
            if not candidates:
                break
            block_id = candidates[0]
            assert self.block_mgr.is_cache_owned(block_id), (
                f"cache entry {block_id} lost ownership (hash {h:#x})"
            )
            self.block_mgr.retain(block_id)
            self._lru.move_to_end(block_id)  # touch（更新 LRU 新鲜度）
            hit_blocks.append(block_id)
            prev_hash = h
        if hit_blocks:
            self._hits += 1
        return hit_blocks, len(hit_blocks) * self.block_size

    # ------------------------------------------------------------------ insert

    def insert_block(self, token_ids: Sequence[int], block_index: int, block_id: int) -> bool:
        """插入序列第 block_index 个完整块（token 范围 [i*bs, (i+1)*bs)）。

        哈希链需要前缀哈希：由调用方保证 0..block_index-1 的块已注册。
        重复内容的块（hash 已存在）不重复注册，返回 False。
        调用前提：block_mgr 中该 block ref >= 1（mark_cached 需要 live 块）。
        """
        start = block_index * self.block_size
        chunk = token_ids[start : start + self.block_size]
        assert len(chunk) == self.block_size, "insert_block: 非完整块"
        prev_hash = self._chain_hash(token_ids, block_index)
        h = self.hash_block(prev_hash, chunk)
        if h in self._by_hash:
            return False  # 相同内容已在缓存，本块保持私有（不入 cache-owned 集）
        if block_id in self._block_hash:
            raise ValueError(f"block {block_id} already registered with hash {self._block_hash[block_id]:#x}")
        self._by_hash.setdefault(h, deque()).append(block_id)
        self._block_hash[block_id] = h
        self._lru[block_id] = h  # 最新鲜
        self.block_mgr.mark_cached(block_id)
        return True

    def insert_full_blocks(self, token_ids: Sequence[int], block_table: list[int], num_blocks: int) -> int:
        """把 block_table 的前 num_blocks 个（跳过已缓存命中的部分由调用方控制
        num_blocks 起点实现）注册进缓存。返回成功注册数。"""
        inserted = 0
        for i in range(num_blocks):
            if i >= len(block_table):
                break
            if self.insert_block(token_ids, i, block_table[i]):
                inserted += 1
        return inserted

    # ------------------------------------------------------------------ evict

    def evict(self, need: int) -> int:
        """从 LRU 最旧端淘汰 ref==0 的 cache-owned 块，至多 need 个。

        被 pin 的块（ref>0）不可淘汰：跳过它们继续向新方向扫描。
        Returns: 实际淘汰数。
        """
        evicted = 0
        while evicted < need:
            victim: tuple[int, int] | None = None
            for block_id, h in self._lru.items():  # OrderedDict 按插入序 = LRU 旧→新
                if self.block_mgr.ref_count(block_id) == 0:
                    victim = (block_id, h)
                    break
            if victim is None:
                break
            block_id, h = victim
            self._lru.pop(block_id)
            self._by_hash[h].remove(block_id)
            if not self._by_hash[h]:
                del self._by_hash[h]
            del self._block_hash[block_id]
            self.block_mgr.unmark_cached(block_id)  # ref==0 → 回收进 free 池
            evicted += 1
        if evicted:
            _log.debug("evicted %d prefix blocks (need=%d)", evicted, need)
        return evicted

    # ------------------------------------------------------------------ release

    def on_request_finished(self, block_table: list[int]) -> None:
        """请求结束：release 其全部块。ref 归零的 cache-owned 块留在缓存中待逐出。"""
        for block_id in block_table:
            self.block_mgr.release(block_id)

    # ------------------------------------------------------------------ stats

    @property
    def hit_rate(self) -> float:
        return self._hits / self._queries if self._queries else 0.0

    @property
    def num_cached_blocks(self) -> int:
        return len(self._block_hash)

    def _chain_hash(self, token_ids: Sequence[int], block_index: int) -> int:
        """计算第 block_index 块的前缀哈希（重算链）。仅供 insert_block。

        注意：match 命中的前缀哈希缓存在 Sequence 侧时可直接传递（Phase 2 优化）；
        CPU 阶段重算可接受。
        """
        prev_hash = 0
        for i in range(block_index):
            chunk = token_ids[i * self.block_size : (i + 1) * self.block_size]
            prev_hash = self.hash_block(prev_hash, chunk)
        return prev_hash

    def debug_check_consistency(self) -> None:
        """_by_hash / _lru / _block_hash 三表互恰。"""
        all_cached = set(self._block_hash)
        assert set(self._lru) == all_cached, "LRU 与注册表不一致"
        flat = [b for dq in self._by_hash.values() for b in dq]
        assert set(flat) == all_cached, "_by_hash 与注册表不一致"
        assert len(flat) == len(set(flat)), "_by_hash 出现重复块"
        for b, h in self._block_hash.items():
            assert self.block_mgr.is_cache_owned(b)
            assert b in self._by_hash.get(h, ()), f"块 {b} 不在其哈希桶 {h:#x}"


__all__ = ["PrefixCache"]

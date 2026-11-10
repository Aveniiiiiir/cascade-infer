"""KV block allocator with reference counting.

Mirrors DeepInfer's `csrc/engine/block_manager.hpp` (`Blocks` + `BlockManager`):

- Block id 0 is reserved as the empty/padding block and is never allocated,
  so GPU kernels can address table row 0 as a no-op slot.
- Every block carries a ref count supporting pin/unpin (prefix sharing, fork).
- A block may additionally be *owned by the prefix cache*: when its ref count
  drops to 0 it stays allocated (not returned to the free list) until the
  cache evicts it. This is the ownership model behind "cached blocks are
  evictable, pinned blocks are not".

All operations are O(1). This class is intentionally free of torch/GPU
dependencies so the scheduler stack is testable on Windows.
"""

from __future__ import annotations

from cascade.common.logger import get_logger

_log = get_logger("core.block_manager")


class BlockAllocationError(RuntimeError):
    """Raised when the KV pool is exhausted (triggers scheduler preemption)."""


class BlockManager:
    __slots__ = ("num_blocks", "_ref_counts", "_cache_owned", "_free_blocks", "_num_allocated_high_water")

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.num_blocks = num_blocks
        # 下标 0 保留为空块；可用块 id 为 1..num_blocks
        self._ref_counts = [0] * (num_blocks + 1)
        self._cache_owned = [False] * (num_blocks + 1)
        self._free_blocks = list(range(num_blocks, 0, -1))  # 栈式 LIFO
        self._num_allocated_high_water = 0

    # ------------------------------------------------------------------ alloc

    def allocate(self) -> int:
        """Allocate one block with ref count 1. Raises on exhaustion."""
        if not self._free_blocks:
            raise BlockAllocationError("KV block pool exhausted")
        block_id = self._free_blocks.pop()
        assert self._ref_counts[block_id] == 0 and not self._cache_owned[block_id]
        self._ref_counts[block_id] = 1
        self._num_allocated_high_water = max(
            self._num_allocated_high_water, self.num_blocks - len(self._free_blocks)
        )
        return block_id

    def try_allocate(self) -> int | None:
        try:
            return self.allocate()
        except BlockAllocationError:
            return None

    # ------------------------------------------------------------------ refs

    def retain(self, block_id: int) -> None:
        """Pin: increment the ref count (prefix-cache hit / fork)."""
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0 and not self._cache_owned[block_id]:
            raise ValueError(f"retain on free block {block_id}")
        self._ref_counts[block_id] += 1

    def release(self, block_id: int) -> None:
        """Unpin: decrement ref count. A ref-0 cache-owned block stays held
        by the cache until eviction; a ref-0 plain block returns to the pool."""
        self._check_id(block_id)
        if self._ref_counts[block_id] <= 0:
            raise ValueError(f"release on block {block_id} with ref {self._ref_counts[block_id]}")
        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id] == 0 and not self._cache_owned[block_id]:
            self._free_blocks.append(block_id)

    def ref_count(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._ref_counts[block_id]

    # ------------------------------------------------------------------ cache ownership

    def mark_cached(self, block_id: int) -> None:
        """Prefix cache takes ownership; block stays allocated at ref==0."""
        self._check_id(block_id)
        if self._ref_counts[block_id] <= 0:
            raise ValueError(f"mark_cached on non-live block {block_id}")
        assert not self._cache_owned[block_id], f"block {block_id} already cache-owned"
        self._cache_owned[block_id] = True

    def unmark_cached(self, block_id: int) -> None:
        """Evict from cache ownership; ref-0 blocks return to the free pool."""
        self._check_id(block_id)
        if not self._cache_owned[block_id]:
            raise ValueError(f"unmark_cached on non-cache block {block_id}")
        self._cache_owned[block_id] = False
        if self._ref_counts[block_id] == 0:
            self._free_blocks.append(block_id)

    def is_cache_owned(self, block_id: int) -> bool:
        return self._cache_owned[block_id]

    # ------------------------------------------------------------------ stats

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    @property
    def num_evictable_blocks(self) -> int:
        """ref==0 且 cache-owned：可被前缀缓存逐出的块数。"""
        return sum(
            1
            for i in range(1, self.num_blocks + 1)
            if self._cache_owned[i] and self._ref_counts[i] == 0
        )

    @property
    def high_water_mark(self) -> int:
        return self._num_allocated_high_water

    # ------------------------------------------------------------------ debug

    def debug_check_consistency(self) -> None:
        """Fuzz-test invariant: free list ↔ (ref==0 且非 cache-owned) 完全一致。"""
        expected_free = {
            i
            for i in range(1, self.num_blocks + 1)
            if self._ref_counts[i] == 0 and not self._cache_owned[i]
        }
        assert set(self._free_blocks) == expected_free, (
            f"free list mismatch: free={sorted(self._free_blocks)} "
            f"expected={sorted(expected_free)}"
        )
        assert len(self._free_blocks) == len(set(self._free_blocks)), "duplicate in free list"
        assert all(self._ref_counts[i] >= 0 for i in range(self.num_blocks + 1))
        # cache-owned 块必然曾经 live：ref 可为 0 但不能是"从未分配"
        for i in range(1, self.num_blocks + 1):
            if self._cache_owned[i]:
                assert i not in set(self._free_blocks)

    def _check_id(self, block_id: int) -> None:
        if not (1 <= block_id <= self.num_blocks):
            raise ValueError(f"invalid block id {block_id}")


__all__ = ["BlockManager", "BlockAllocationError"]

"""CUDA Graph runner：decode forward 的图捕获与回放（Phase 4 核心交付）。

结构（对齐 DeepInfer cuda_graph_runner.py / vLLM）：
- 档位按 **n_seqs（decode 行数）** 分档：2 的幂到 interval(8) 后等差到
  max_batch_size；
- 全档位**共享同一 memory pool**（torch.cuda.graph_pool_handle）；
- 捕获内容 = `model.forward_hidden`（transformer 主体，不含 lm_head——
  每档位 logits [T, vocab] 太大，lm_head 的一次 GEMM 留图外）；
- 输入缓冲持久化、回放前**原地写入**（input_ids / positions /
  block_tables 真实行；padding 行恒指向 scratch 块，写入无害）；
- 图捕获安全三要素：mla 纯 decode 路径无 .item()（seqlens=positions+1）、
  decode 注意力 num_splits 固定、MoE pad_to 固定（moe_align 尺寸与 grid
  均由形状决定）。

批不变性：padding 行与真实行在 attention/MoE 中按行独立（tl.dot 行间
无交互；MoE 分桶顺序由 stable-argsort 决定，垃圾 padding token 只改变
块内布局不改变行结果）⇒ 图回放与 eager bit 级一致。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import torch

from cascade.common.env import envs
from cascade.common.logger import get_logger
from cascade.config import EngineConfig
from cascade.worker.cache_engine import CacheEngine

_log = get_logger("worker.cuda_graph_runner")


@dataclass(slots=True)
class DecodeGraphTier:
    size: int
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor  # [T] int64（回放前原地写真实行）
    positions: torch.Tensor  # [T] int64
    seq_ids: torch.Tensor  # [T] int64（恒 arange，无需重写）
    block_tables: torch.Tensor  # [T, max_bt] int64（真实行重写；padding 行=scratch）
    hidden: torch.Tensor  # [T, H] 捕获期分配的输出缓冲


class CudaGraphRunner:
    __slots__ = (
        "model",
        "cache",
        "cfg",
        "device",
        "max_batch_size",
        "num_splits",
        "scratch_block",
        "block_size",
        "max_bt",
        "_tiers",
        "_tier_sizes",
    )

    def __init__(
        self,
        model,  # DeepseekV2ForCausalLM（bf16 on cuda）
        cache: CacheEngine,
        cfg: EngineConfig,
        device: str,
        max_batch_size: int,
        scratch_block: int,
        num_splits: int | None = None,
    ) -> None:
        assert str(device).startswith("cuda")
        self.model = model
        self.cache = cache
        self.cfg = cfg
        self.device = device
        self.max_batch_size = max_batch_size
        self.num_splits = num_splits if num_splits is not None else envs.CASCADE_CUDAGRAPH_NUM_SPLITS
        self.scratch_block = scratch_block
        self.block_size = cache.cache_cfg.block_size
        # block 表列数：覆盖 runtime_max_model_len + 投机余量
        self.max_bt = cfg.runtime_max_model_len // self.block_size + 2
        self._tiers: dict[int, DecodeGraphTier] = {}
        self._tier_sizes: list[int] = []
        for size in self._build_tier_sizes(max_batch_size):
            self._tiers[size] = self._capture_tier(size)
            self._tier_sizes.append(size)
        _log.info(
            "CUDA graphs captured: %s (num_splits=%d, scratch_block=%d)",
            self._tier_sizes,
            self.num_splits,
            self.scratch_block,
        )

    # ------------------------------------------------------------------ tiers

    def _build_tier_sizes(self, max_batch: int) -> list[int]:
        interval = envs.CASCADE_CUDAGRAPH_BATCH_SIZE_INTERVAL
        sizes = [s for s in (1, 2, 4, 8) if s <= max_batch]
        sizes += list(range(max(16, interval), max_batch + 1, interval))
        if not sizes or sizes[-1] != max_batch:
            sizes.append(max_batch)
        return sorted(set(sizes))

    def can_run(self, n_seqs: int) -> bool:
        return bool(self._tiers) and n_seqs <= self.max_batch_size

    def _select(self, n: int) -> DecodeGraphTier:
        idx = bisect_left(self._tier_sizes, n)
        size = self._tier_sizes[min(idx, len(self._tier_sizes) - 1)]
        return self._tiers[size]

    # ------------------------------------------------------------------ capture

    def _capture_tier(self, T: int) -> DecodeGraphTier:
        dev = self.device
        input_ids = torch.zeros(T, dtype=torch.int64, device=dev)
        positions = torch.zeros(T, dtype=torch.int64, device=dev)
        seq_ids = torch.arange(T, dtype=torch.int64, device=dev)
        # 全部行初始指向 scratch 块（warmup/捕获的写入无害）
        block_tables = torch.full(
            (T, self.max_bt), self.scratch_block, dtype=torch.int64, device=dev
        )
        segment_lens = [1] * T

        def callable_() -> torch.Tensor:
            return self.model.forward_hidden(
                input_ids, positions, seq_ids, segment_lens, block_tables,
                self.cache, num_splits=self.num_splits,
            )

        # MoE 图模式：本档位的 pad_to（moe_align 固定尺寸；仅捕获期生效）
        moe_pad_to = self._set_moe_graph_pad_to(T)
        try:
            # warmup（side stream，torch.cuda.graph 要求先跑几遍）
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    callable_()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            pool = self._graph_pool()
            with torch.cuda.graph(graph, pool=pool):
                hidden = callable_()
            return DecodeGraphTier(
                size=T, graph=graph, input_ids=input_ids, positions=positions,
                seq_ids=seq_ids, block_tables=block_tables, hidden=hidden,
            )
        finally:
            self._clear_moe_graph_pad_to()

    _pool_handle: object | None = None

    def _graph_pool(self):
        if CudaGraphRunner._pool_handle is None:
            CudaGraphRunner._pool_handle = torch.cuda.graph_pool_handle()
        return CudaGraphRunner._pool_handle

    def _set_moe_graph_pad_to(self, T: int) -> int:
        from cascade.model_executor.layers.moe import DeepseekMoE

        K = self.cfg.model.num_experts_per_tok
        E = self.cfg.model.n_routed_experts
        bm = 16  # fused_moe 默认 BLOCK_M
        pad_to = T * K + E * bm  # 最坏填充上界
        n = 0
        for m in self.model.modules():
            if isinstance(m, DeepseekMoE):
                m.graph_pad_to = pad_to
                n += 1
        return n

    def _clear_moe_graph_pad_to(self) -> None:
        from cascade.model_executor.layers.moe import DeepseekMoE

        for m in self.model.modules():
            if isinstance(m, DeepseekMoE):
                m.graph_pad_to = None

    # ------------------------------------------------------------------ replay

    def replay(
        self,
        tokens: list[int],  # 真实 decode 行的输入 token（长度 n）
        positions: list[int],  # 真实行的绝对位置
        block_rows: list[list[int]],  # 真实行的 block 表
    ) -> torch.Tensor:
        """原地写输入 → 回放 → hidden [T, H]（前 n 行为真实结果）。"""
        n = len(tokens)
        assert n <= self.max_batch_size
        tier = self._select(n)
        dev = self.device
        tier.input_ids[:n].copy_(torch.tensor(tokens, dtype=torch.int64))
        tier.positions[:n].copy_(torch.tensor(positions, dtype=torch.int64))
        bt = torch.zeros(n, self.max_bt, dtype=torch.int64)
        for i, row in enumerate(block_rows):
            bt[i, : len(row)] = torch.as_tensor(row, dtype=torch.int64)
        tier.block_tables[:n].copy_(bt.to(dev))
        tier.graph.replay()
        return tier.hidden


__all__ = ["CudaGraphRunner", "DecodeGraphTier"]

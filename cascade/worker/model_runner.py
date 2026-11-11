"""Model runner: SchedulerOutput → GPU forward → sampled tokens (sync Phase 2)。

三段式（forward → post_process(prev) → sample）的 ZOS 版本在 Phase 4 重构；
本文件先把闭环打通：prepare_batch 张量构建（对应 DeepInfer
`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp` 的 Python 参考版）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cascade.common.logger import get_logger
from cascade.config import CacheConfig, EngineConfig, ModelConfig
from cascade.core.scheduler import SchedulerOutput
from cascade.model_executor.model_loader import load_deepseek_v2
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.sampling.sampler import Sampler, build_sampling_rows
from cascade.worker.cache_engine import CacheEngine
from cascade.worker.kv_layout import derive_num_blocks

_log = get_logger("worker.model_runner")


@dataclass(slots=True)
class ForwardBatch:
    """展平 varlen 批的张量化形态（段 = 一个 scheduled seq 的本步 token）。"""

    input_ids: torch.Tensor  # [T]
    positions: torch.Tensor  # [T] 绝对位置
    seq_ids: torch.Tensor  # [T] → block_tables 行
    segment_lens: list[int]  # 每段 token 数（与 scheduled 顺序一致）
    block_tables: torch.Tensor  # [n_seqs, max_blocks]
    #: 每 scheduled seq 是否在本步产出采样 token（decode 或 prefill 收尾）
    sample_flags: list[bool]
    #: 每段在展平批中的起始偏移（供采样行定位）
    segment_offsets: list[int]


@dataclass(slots=True)
class SampleLaunch:
    """一步采样的在途状态（ZOS：launch → 异步 D2H → 下一步 post_process 提交）。"""

    request_ids: list[str]
    pinned: torch.Tensor  # [R] int32 CPU（pinned；CUDA 事件守卫）
    event: object | None  # torch.cuda.Event（无 CUDA 时 None）
    sched_out: "SchedulerOutput"  # 提交时需要（持有以简化引擎侧簿记）

    def wait_tokens(self) -> dict[str, list[int]]:
        """唯一 CPU-GPU 同步点：等待采样 + D2H 完成 → request_id → token。"""
        if self.event is not None:
            self.event.synchronize()
        tokens = self.pinned.tolist()
        return {rid: [int(t)] for rid, t in zip(self.request_ids, tokens)}


class ModelRunner:
    __slots__ = ("engine_cfg", "model", "cfg", "cache", "device", "dtype", "use_absorbed", "sampler")

    def __init__(
        self,
        engine_cfg: EngineConfig,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_blocks: int | None = None,
        use_absorbed: bool = True,
    ) -> None:
        self.engine_cfg = engine_cfg
        self.device = device
        self.dtype = dtype
        self.use_absorbed = use_absorbed
        self.sampler = Sampler()
        self.model, self.cfg = load_deepseek_v2(model_dir, device=device, dtype=dtype)
        if num_blocks is None:
            num_blocks = self._profile_num_blocks(engine_cfg.cache)
        self.cache = CacheEngine(self.cfg, engine_cfg.cache, num_blocks, device=device, dtype=dtype)
        _log.info("ModelRunner ready: %d KV blocks (bs=%d)", num_blocks, engine_cfg.cache.block_size)

    def _profile_num_blocks(self, cache_cfg: CacheConfig) -> int:
        """显存预算推 block 数。真实 profile run（峰值测量）Phase 2 集群验收时替换。"""
        if self.device.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info()
            budget = int(free * cache_cfg.gpu_memory_utilization)
        else:
            budget = 256 * 2**20  # CPU 测试默认 256 MiB
        n = derive_num_blocks(budget, self.cfg, cache_cfg.block_size)
        _log.info("KV budget %.2f GiB → %d blocks", budget / 2**30, n)
        return n

    # ------------------------------------------------------------------ batch prep

    def prepare_batch(self, sched_out: SchedulerOutput) -> ForwardBatch:
        """SchedulerOutput → ForwardBatch。

        注意乐观推进：schedule() 已把 num_tokens_computed 前移 num_new_tokens，
        本步要喂的 token 范围是 [computed - num_new, computed)。
        """
        dev = self.device
        input_ids: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        seq_ids: list[torch.Tensor] = []
        segment_lens: list[int] = []
        sample_flags: list[bool] = []
        max_blocks = 1
        for row, s in enumerate(sched_out.scheduled):
            seq = s.seq
            start = seq.num_tokens_computed - s.num_new_tokens
            toks = [seq.token_at(p) for p in range(start, start + s.num_new_tokens)]
            input_ids.append(torch.tensor(toks, dtype=torch.int64, device=dev))
            positions.append(
                torch.arange(start, start + s.num_new_tokens, dtype=torch.int64, device=dev)
            )
            seq_ids.append(torch.full((s.num_new_tokens,), row, dtype=torch.int64, device=dev))
            segment_lens.append(s.num_new_tokens)
            sample_flags.append((not s.is_prefill) or s.completes_prefill)
            max_blocks = max(max_blocks, len(seq.block_table))
        n_seqs = len(sched_out.scheduled)
        block_tables = torch.zeros(n_seqs, max_blocks, dtype=torch.int64, device=dev)
        for row, s in enumerate(sched_out.scheduled):
            bt = s.seq.block_table
            if bt:
                block_tables[row, : len(bt)] = torch.tensor(bt, dtype=torch.int64, device=dev)
        offsets = []
        acc = 0
        for n in segment_lens:
            offsets.append(acc)
            acc += n
        return ForwardBatch(
            input_ids=torch.cat(input_ids) if input_ids else torch.zeros(0, dtype=torch.int64, device=dev),
            positions=torch.cat(positions) if positions else torch.zeros(0, dtype=torch.int64, device=dev),
            seq_ids=torch.cat(seq_ids) if seq_ids else torch.zeros(0, dtype=torch.int64, device=dev),
            segment_lens=segment_lens,
            block_tables=block_tables,
            sample_flags=sample_flags,
            segment_offsets=offsets,
        )

    # ------------------------------------------------------------------ forward + sample

    def forward(self, batch: ForwardBatch) -> torch.Tensor:
        return self.model(
            batch.input_ids,
            batch.positions,
            batch.seq_ids,
            batch.segment_lens,
            batch.block_tables,
            self.cache,
            use_absorbed=self.use_absorbed,
        )

    def launch_sample(
        self,
        logits: torch.Tensor,
        sched_out: SchedulerOutput,
        batch: ForwardBatch,
    ) -> SampleLaunch | None:
        """三段式第 3 段：启动确定性采样 + 异步 D2H（不碰 CPU 结果）。

        token 的绝对位置 = seq.num_tokens（update 追加前），
        golden 索引 = seq.num_output_tokens（追加前）。
        """
        sel: list[tuple[str, object, int, int]] = []
        rows_idx: list[int] = []
        for s, flag, off, n in zip(
            sched_out.scheduled, batch.sample_flags, batch.segment_offsets, batch.segment_lens
        ):
            if not flag:
                continue
            seq = s.seq
            rows_idx.append(off + n - 1)
            sel.append(
                (
                    seq.request_id,
                    seq.sampling,
                    seq.num_tokens,
                    seq.sampling.golden_at(seq.num_output_tokens),
                )
            )
        if not sel:
            return None
        rows = build_sampling_rows(sel, self.device)
        logits_rows = logits[rows_idx].to(torch.float32).contiguous()
        tokens = self.sampler.sample_tensors(logits_rows, rows)
        is_cuda = str(self.device).startswith("cuda")
        pinned = torch.empty(
            tokens.shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=is_cuda,
        )
        pinned.copy_(tokens, non_blocking=is_cuda)
        event = None
        if is_cuda:
            event = torch.cuda.Event()
            event.record()
        return SampleLaunch(
            request_ids=[rid for rid, _, _, _ in sel],
            pinned=pinned,
            event=event,
            sched_out=sched_out,
        )


__all__ = ["ModelRunner", "ForwardBatch", "SampleLaunch"]

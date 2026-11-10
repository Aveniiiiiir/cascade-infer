"""Scheduler v1: continuous batching with chunked prefill (CPU side).

Semantics mirror DeepInfer's `csrc/engine/scheduler.hpp` where applicable:

- Three queues: ``waiting`` (prompt 未算完/待重算) → ``running`` (decode)。
  DeepInfer has an intermediate ``prompted`` queue for prefill-completed
  sequences; we fold it into the waiting→running transition instead.
- Prefill: FCFS from waiting, budget-capped. **Multiple whole prompts are
  packed into one step** (bounded by max_num_batched_tokens / sum-K /
  max_num_seqs); an over-budget prompt is chunked (截断留队头) and at most
  ONE chunked sequence runs per step (DeepInfer :543 breaks only on the
  chunked case too).
- Decode: LCFS — schedule newest first (ascending output_len); preemption
  targets the longest-output sequence first (DeepInfer: 抢占队头).
- Preemption: RECOMPUTE mode resets the sequence for recomputation (cheap
  with prefix caching — its blocks stay in the cache); STOP mode finishes it.
  Structured-output sequences cannot be recomputed → preemption kills them.
- **Never-fit guard**: a sequence whose KV footprint can never fit the pool
  finishes as FINISHED_IGNORED (DeepInfer's NEVER path, scheduler.hpp:389) —
  the engine must not livelock on unsatisfiable growth.
- Prefix matching clamps the hit so at least one token is always computed
  fresh (vLLM V1 / DeepInfer fork clamp `set_num_tokens(min(..., len-1))`):
  a fully-cached sequence would otherwise produce a 0-token prefill row.
- **Optimistic advancement**: `num_tokens_computed` is advanced at *schedule*
  time, before the step executes — the same one-step-lag protocol DeepInfer's
  ZOS builds on, here in its trivially-consistent synchronous form. Phase 4
  (ZOS) keeps this protocol and only shifts *when* update_from_output runs.

Divergence from DeepInfer (documented): we mix decode tokens and prefill
chunks in a single varlen batch (vLLM V1 style) instead of alternating
whole-engine prefill/decode step types. DeepInfer's step-type separation
serves DP consensus voting and EPD disaggregation, both Phase 6+ concerns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from cascade.common.logger import get_logger
from cascade.config import EngineConfig
from cascade.core.block_manager import BlockAllocationError, BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.sequence import (
    RequestOutput,
    Sequence,
    SequenceStatus,
)
from cascade.sampling_params import SamplingParams

_log = get_logger("core.scheduler")


@dataclass(slots=True)
class ScheduledSeq:
    """Per-step scheduling decision for one sequence (POD snapshot)."""

    seq: Sequence
    num_new_tokens: int  # 本步要计算 KV 的 token 数（prefill chunk 或 decode 的 1）
    is_prefill: bool
    completes_prefill: bool = False  # prefill 末 chunk：执行后将产出采样 token


@dataclass(slots=True)
class SchedulerOutput:
    step_id: int
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    total_new_tokens: int = 0
    num_decode_seqs: int = 0
    #: COW 计划：(dst_block, src_block)。GPU 执行器需在 forward 前做 KV 拷贝
    #: （Phase 2 消费；CPU/mock 执行器忽略）。
    cow_copies: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    __slots__ = (
        "cfg",
        "block_size",
        "block_mgr",
        "prefix_cache",
        "waiting",
        "running",
        "_step",
        "_last_output",
        "_pending_outputs",
        "_preempted_this_step",
        "_trace",
    )

    def __init__(
        self,
        cfg: EngineConfig,
        block_mgr: BlockManager,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.block_size = cfg.cache.block_size
        self.block_mgr = block_mgr
        self.prefix_cache = prefix_cache if cfg.scheduler.enable_prefix_caching else None
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._step = 0
        self._last_output: SchedulerOutput | None = None
        self._pending_outputs: list[RequestOutput] = []
        #: 本步被抢占的 seq（含 STOP/RECOMPUTE），调度循环内跳过
        self._preempted_this_step: set[str] = set()
        self._trace: list[str] = []

    # ------------------------------------------------------------------ intake

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        has_so: bool = False,
    ) -> Sequence:
        """入队。SamplingParams 立即捕获为 POD 快照（DeepInfer 语义：
        调度热路径不再触碰原始参数对象）。"""
        seq = Sequence(request_id, prompt_token_ids, sampling_params.snapshot(), has_so=has_so)
        # 上限校验：即使无任何前缀命中，prompt 也必须放得进 KV 池
        min_blocks = seq.num_blocks_for(len(prompt_token_ids), self.block_size)
        if min_blocks > self.block_mgr.num_blocks:
            raise ValueError(
                f"request {request_id}: prompt of {len(prompt_token_ids)} tokens needs "
                f"{min_blocks} blocks but the pool has {self.block_mgr.num_blocks}"
            )
        if seq.num_tokens + 1 > self.cfg.runtime_max_model_len:
            raise ValueError(
                f"request {request_id}: prompt longer than max_model_len={self.cfg.model.max_model_len}"
            )
        self.waiting.append(seq)
        return seq

    @property
    def has_unfinished_requests(self) -> bool:
        """含 schedule→update 窗口内"队列外悬空"的 seq（完成 prefill 待提交者）。"""
        if self.waiting or self.running:
            return True
        if self._last_output is not None:
            return any(not s.seq.is_finished for s in self._last_output.scheduled)
        return False

    def abort(self, request_id: str) -> bool:
        seq = self._find(request_id)
        if seq is None or seq.is_finished:
            return False
        self._finish_sequence(seq, SequenceStatus.FINISHED_ABORTED)
        return True

    # ------------------------------------------------------------------ schedule

    def schedule(self) -> SchedulerOutput:
        """调度一步：decode 行优先（延迟敏感），再用剩余预算排 prefill。

        乐观推进：本函数内即推进 num_tokens_computed 并分配 KV 块；
        update_from_output 在执行结果返回后做提交（追加 token/判停）。
        """
        assert self._last_output is None, (
            "schedule() 被重入：上一步的 update_from_output 尚未调用 "
            "（乐观推进会被覆盖导致状态腐蚀；ZOS Phase 4 将改为待提交队列）"
        )
        out = SchedulerOutput(step_id=self._step + 1)
        self._preempted_this_step = set()

        budget = self.cfg.scheduler.max_num_batched_tokens
        max_seqs = self.cfg.scheduler.max_num_seqs
        sum_seqlen_k = 0

        # ---- decode（LCFS：最新请求优先 = output_len 升序）----
        for seq in sorted(self.running, key=lambda s: s.num_output_tokens):
            if len(out.scheduled) >= max_seqs or budget <= 0:
                break
            if seq.is_finished or seq.request_id in self._preempted_this_step:
                continue  # 排序副本中的已抢占/已结束 seq
            ok = self._acquire_kv(seq, num_new_tokens=1, out=out)
            if not ok:
                # 永不满足：自身足迹已超池（自抢占排除自身 → 无候选可抢）。
                # DeepInfer NEVER 路径：判 FINISHED_IGNORED 而非活锁空转。
                if seq.num_blocks_for(seq.num_tokens, self.block_size) > self.block_mgr.num_blocks:
                    _log.warning(
                        "request %s: KV footprint %d blocks exceeds pool %d → FINISHED_IGNORED",
                        seq.request_id,
                        seq.num_blocks_for(seq.num_tokens, self.block_size),
                        self.block_mgr.num_blocks,
                    )
                    self._finish_sequence(seq, SequenceStatus.FINISHED_IGNORED)
                continue
            out.scheduled.append(ScheduledSeq(seq, 1, is_prefill=False))
            seq.num_tokens_computed += 1
            out.total_new_tokens += 1
            out.num_decode_seqs += 1
            budget -= 1

        # ---- prefill（FCFS；多个完整 prompt 可打包进同一步，sum-K/max_seqs 封顶；
        #       超预算者 chunk 化，一步至多一个 chunked seq —— DeepInfer :543 只在
        #       chunked 情形 break，完整 prompt 继续装）----
        chunked_seen = False
        while self.waiting and budget > 0 and len(out.scheduled) < max_seqs and not chunked_seen:
            seq = self.waiting.popleft()
            if seq.is_finished:  # abort/ignore 的 seq 残留在 waiting
                continue
            if seq.request_id in self._preempted_this_step:
                # 本步被抢占重排进 waiting 的 seq：留待下一步
                self.waiting.appendleft(seq)
                break
            # 前缀匹配先于 chunk 计算：命中会推进 computed（命中部分 KV 已存在），
            # chunk 只需覆盖剩余部分 —— 顺序颠倒会造成命中长度双重计数。
            # 匹配结果 clamp 到至少留 1 个新 token（全量命中会产生 0-token 行，
            # 执行器无法从中采样 —— vLLM V1 / DeepInfer fork clamp 同语义）。
            self._try_match(seq)
            remaining = seq.num_tokens - seq.num_tokens_computed
            assert remaining >= 1, (
                f"{seq.request_id}: 非法的调度状态 computed={seq.num_tokens_computed} "
                f"num_tokens={seq.num_tokens}（fork/回滚状态腐蚀）"
            )
            chunk = min(remaining, budget)
            if out.scheduled and sum_seqlen_k + seq.num_tokens > self.cfg.scheduler.prefill_sum_seqlen_k:
                # sum-K 预算（DeepInfer prefill_max_sum_seqlen_k）：放不下就留队头
                self.waiting.appendleft(seq)
                break
            completes = chunk >= remaining
            ok = self._acquire_kv(seq, num_new_tokens=chunk, out=out)
            if not ok:
                # 永不满足：重算后的完整足迹超池（prompt 之外还要装回已生成输出）
                if seq.num_blocks_for(seq.num_tokens, self.block_size) > self.block_mgr.num_blocks:
                    self._rollback_match(seq, out)
                    _log.warning(
                        "request %s: recompute footprint exceeds pool → FINISHED_IGNORED",
                        seq.request_id,
                    )
                    self._finish_sequence(seq, SequenceStatus.FINISHED_IGNORED)
                    continue
                # 分配失败（含抢占后仍不足）：回滚匹配、放回队头
                self._rollback_match(seq, out)
                self.waiting.appendleft(seq)
                chunked_seen = True  # 避免同一步反复尝试
                continue
            out.scheduled.append(ScheduledSeq(seq, chunk, is_prefill=True, completes_prefill=completes))
            seq.num_tokens_computed += chunk
            out.total_new_tokens += chunk
            budget -= chunk
            sum_seqlen_k += seq.num_tokens
            if completes:
                # 末 chunk：seq 暂存于 scheduled，update 时 waiting→running；
                # 继续尝试打包下一个完整 prompt（budget 允许时）
                pass
            else:
                self.waiting.appendleft(seq)  # chunked：留队头，本步不再收第二个 chunked
                chunked_seen = True

        assert out.total_new_tokens == sum(s.num_new_tokens for s in out.scheduled)
        self._step = out.step_id
        self._last_output = out
        self._trace.append(self._trace_line(out))
        return out

    # ------------------------------------------------------------------ commit

    def update_from_output(self, sampled: dict[str, list[int]]) -> list[RequestOutput]:
        """提交一步的采样结果（ZOS Phase 4 起，本函数将滞后一步调用）。

        窗口内被 abort/判死的 seq 直接跳过（DeepInfer process_outputs :906
        对 finished 跳过，不靠 assert）——其块已在 _finish_sequence 释放。

        Args:
            sampled: request_id → 新采样 token 列表。prefill 中间 chunk 无条目；
                末 chunk 与 decode 各产出 1 个 token（Phase 5 MTP 起可为多个）。
        """
        outputs = list(self._pending_outputs)
        self._pending_outputs.clear()
        last = self._last_output
        if last is None:
            return outputs
        for scheduled in last.scheduled:
            seq = scheduled.seq
            if seq.is_finished:
                continue  # 窗口内被 abort/抢占判死：丢弃本步采样，不追加
            new_tokens = sampled.get(seq.request_id, [])
            if scheduled.is_prefill and not scheduled.completes_prefill:
                assert not new_tokens, f"{seq.request_id}: 中间 chunk 不应产出 token"
                continue
            assert len(new_tokens) == 1, (
                f"{seq.request_id}: Phase 1 每步至多 1 个 token, got {new_tokens}"
            )
            token = new_tokens[0]
            seq.append_output_tokens([token])

            # ---- 判停（首个 stop token 截断；MTP 多 token 时同样适用）----
            finish_status = self._check_stop(seq, token)
            if finish_status is not None:
                seq.finish(finish_status)
            # prefill 收尾：waiting → running（完成的留在 scheduled 里直到下一步）
            if scheduled.completes_prefill and not seq.is_finished:
                assert seq.status is SequenceStatus.WAITING
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            # 新满块入前缀缓存
            self._register_full_blocks(seq)
            # ---- 产出（finish_reason 已就位）----
            outputs.append(
                RequestOutput(
                    request_id=seq.request_id,
                    new_token_ids=[token],
                    output_token_ids=list(seq.output_token_ids),
                    finish_reason=seq.finish_reason,
                    num_cached_tokens=seq.num_cached_tokens,
                )
            )
            if seq.is_finished:
                self._release_and_remove(seq)
        self._last_output = None
        return outputs

    # ------------------------------------------------------------------ internals: block accounting

    def _acquire_kv(self, seq: Sequence, num_new_tokens: int, out: SchedulerOutput) -> bool:
        """为 seq 本步新增 num_new_tokens 个 KV token 分配块（乐观推进协议）。

        前缀匹配由调用方（prefill 循环）在计算 chunk 之前完成；
        decode 序列的 computed 恒为 num_tokens-1，无需匹配。
        顺序（审查 #6 修复）：容量检查（含 COW 余量）→ COW 克隆 → 追加分配；
        COW 使用 try_allocate，任何失败都不留半途状态。
        Returns: 是否成功（失败时 seq 状态不变）。
        """
        target = seq.num_blocks_for(seq.num_tokens_computed + num_new_tokens, self.block_size)
        have = len(seq.block_table)
        need = target - have
        assert need >= 0, f"{seq.request_id}: block table {have} 超过目标 {target}"
        cow_needed = self._cow_needed(seq)
        if need == 0 and not cow_needed:
            return True
        # 抢占候选排除：本步已调度的 seq + 当前 seq 自身
        exclude = frozenset(s.seq.request_id for s in out.scheduled) | {seq.request_id}
        if not self._ensure_capacity(need + (1 if cow_needed else 0), exclude):
            return False
        if cow_needed:
            new_block = self.block_mgr.try_allocate()
            if new_block is None:  # pragma: no cover - 容量已确保
                return False
            old_block = seq.block_table[-1]
            seq.block_table[-1] = new_block
            self.block_mgr.release(old_block)
            out.cow_copies.append((new_block, old_block))
        if need == 0:
            return True
        try:
            for _ in range(need):
                seq.block_table.append(self.block_mgr.allocate())
        except BlockAllocationError:  # pragma: no cover - 容量已确保
            return False
        return True

    def _cow_needed(self, seq: Sequence) -> bool:
        """半写尾块被共享（fork 前缀共享）时需要先克隆再写。"""
        if not seq.block_table or seq.num_tokens_computed % self.block_size == 0:
            return False
        return self.block_mgr.ref_count(seq.block_table[-1]) > 1

    def _ensure_capacity(self, need: int, exclude: frozenset[str]) -> bool:
        """确保 can allocate `need` blocks：先逐出缓存，再抢占 running。"""
        free = self.block_mgr.num_free_blocks
        if free >= need:
            return True
        if self.prefix_cache is not None:
            self.prefix_cache.evict(need - free)
            free = self.block_mgr.num_free_blocks
            if free >= need:
                return True
        # 抢占：output 最多者优先（DeepInfer: 抢占队头）
        candidates = [s for s in self.running if s.request_id not in exclude and not s.is_finished]
        candidates.sort(key=lambda s: s.num_output_tokens, reverse=True)
        for victim in candidates:
            if self.block_mgr.num_free_blocks >= need:
                break
            self._preempt(victim)
        return self.block_mgr.num_free_blocks >= need

    def _try_match(self, seq: Sequence) -> bool:
        if self.prefix_cache is None:
            return False
        # 幂等守卫：已匹配（有块且 computed>0）或已有 KV 进度的 seq 不重复匹配，
        # 否则 sum-K 推迟等场景会二次 pin 同样的块
        if seq.block_table or seq.num_tokens_computed > 0:
            return False
        # 匹配对象 = 当前全部 token（prompt + 已生成；RECOMPUTE 后两者都要重算，
        # 前缀命中使重算廉价 —— vLLM V1 语义）
        all_tokens = seq.prompt_token_ids + seq.output_token_ids
        hit_blocks, hit_tokens = self.prefix_cache.match(all_tokens)
        if not hit_blocks:
            return False
        # clamp：至少留 1 个新 token（全量命中 → 0-token prefill 行是执行器
        # 契约破裂；同 vLLM V1 与 DeepInfer fork clamp min(computed, len-1)）
        max_blocks = max(0, (seq.num_tokens - 1) // self.block_size)
        if len(hit_blocks) > max_blocks:
            for block_id in hit_blocks[max_blocks:]:
                self.block_mgr.release(block_id)
            hit_blocks = hit_blocks[:max_blocks]
            hit_tokens = len(hit_blocks) * self.block_size
        if not hit_blocks:
            return False
        seq.block_table = hit_blocks
        seq.num_cached_tokens = hit_tokens
        seq.num_tokens_computed = hit_tokens  # 命中部分 KV 已存在
        return True

    def _rollback_match(self, seq: Sequence, out: SchedulerOutput) -> None:
        """分配失败时回滚本步对 seq 的全部变更（匹配 pin + COW 克隆）。"""
        # COW 回滚：换回旧块、回收新块、撤销执行器拷贝计划
        #（每 seq 每步至多一次 COW：只有半写尾块会被克隆）
        for i in range(len(out.cow_copies) - 1, -1, -1):
            new_block, old_block = out.cow_copies[i]
            if new_block in seq.block_table:
                idx = seq.block_table.index(new_block)
                seq.block_table[idx] = old_block
                self.block_mgr.retain(old_block)
                self.block_mgr.release(new_block)
                out.cow_copies.pop(i)
                break
        if seq.block_table and seq.num_cached_tokens > 0:
            for block_id in seq.block_table:
                self.block_mgr.release(block_id)
            seq.block_table = []
            seq.num_cached_tokens = 0
            seq.num_tokens_computed = 0

    def _register_full_blocks(self, seq: Sequence) -> None:
        """把序列中新变满的块注册进前缀缓存（已有注册的跳过）。"""
        if self.prefix_cache is None or not seq.block_table:
            return
        all_tokens = seq.prompt_token_ids + seq.output_token_ids
        num_full = seq.num_tokens_computed // self.block_size
        for i in range(num_full):
            block_id = seq.block_table[i]
            if self.block_mgr.is_cache_owned(block_id):
                continue
            self.prefix_cache.insert_block(all_tokens, i, block_id)

    # ------------------------------------------------------------------ internals: preemption / finish

    def _preempt(self, victim: Sequence) -> None:
        self._preempted_this_step.add(victim.request_id)
        mode_stop = (
            not self.cfg.scheduler.preemption_recompute
            or victim.has_so  # SO 请求无法重算 → 必判死（DeepInfer 语义）
        )
        if mode_stop:
            self._finish_sequence(victim, SequenceStatus.FINISHED_PREEMPTED)
            return
        # RECOMPUTE：满块先入缓存（重算时可直接命中），再释放全部块
        self._register_full_blocks(victim)
        for block_id in victim.block_table:
            self.block_mgr.release(block_id)
        victim.block_table = []
        victim.num_cached_tokens = 0
        victim.num_tokens_computed = 0
        victim.status = SequenceStatus.WAITING
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        _log.debug("preempted (recompute) %s", victim.request_id)

    def _finish_sequence(self, seq: Sequence, status: SequenceStatus) -> None:
        seq.finish(status)
        self._release_and_remove(seq)
        # FINISHED_* 的最终输出（STOP 抢占 / abort / ignored）
        self._pending_outputs.append(
            RequestOutput(
                request_id=seq.request_id,
                new_token_ids=[],
                output_token_ids=list(seq.output_token_ids),
                finish_reason=seq.finish_reason,
                num_cached_tokens=seq.num_cached_tokens,
            )
        )

    def _release_and_remove(self, seq: Sequence) -> None:
        """释放块并移出所有队列。前缀缓存持有的满块留在缓存（ref 归 0 待逐出）。"""
        for block_id in seq.block_table:
            self.block_mgr.release(block_id)
        seq.block_table = []
        if seq in self.running:
            self.running.remove(seq)
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass

    def _check_stop(self, seq: Sequence, token: int) -> SequenceStatus | None:
        if seq.sampling.is_stop_token(token):
            return SequenceStatus.FINISHED_STOPPED
        if seq.num_output_tokens >= seq.sampling.max_tokens:
            return SequenceStatus.FINISHED_LENGTH_CAPPED
        # 下一个 token 的绝对位置 = num_tokens，必须 < max_model_len
        # （runtime_max_model_len 的投机余量只作用于 KV/位置分配，不放宽输出上限）
        if seq.num_tokens + 1 > self.cfg.model.max_model_len:
            return SequenceStatus.FINISHED_LENGTH_CAPPED
        return None

    # ------------------------------------------------------------------ misc

    def _find(self, request_id: str) -> Sequence | None:
        for seq in list(self.waiting) + self.running:
            if seq.request_id == request_id:
                return seq
        # schedule→update 窗口内的"队列外悬空"seq（完成 prefill 待提交者）
        if self._last_output is not None:
            for s in self._last_output.scheduled:
                if s.seq.request_id == request_id:
                    return s.seq
        return None

    def _trace_line(self, out: SchedulerOutput) -> str:
        parts = [f"step={out.step_id}", f"tok={out.total_new_tokens}", f"dec={out.num_decode_seqs}"]
        for s in out.scheduled:
            parts.append(
                f"{s.seq.request_id}:{'P' if s.is_prefill else 'D'}{s.num_new_tokens}"
                f"{'*' if s.completes_prefill else ''}"
            )
        return " ".join(parts)

    def drain_trace(self) -> list[str]:
        lines = self._trace
        self._trace = []
        return lines


__all__ = ["Scheduler", "SchedulerOutput", "ScheduledSeq"]

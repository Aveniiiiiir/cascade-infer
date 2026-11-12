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
from cascade.config import ModelConfig
from cascade.model_executor.model_loader import load_deepseek_v2
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.sampling.confidence import ConfidenceCalibrator
from cascade.sampling.sampler import Sampler, build_sampling_rows
from cascade.structured_output.matcher import SOMatcher  # noqa: F401  (类型引用)
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
class DraftInfo:
    """上一轮草案（step N 的 sample 阶段发射，step N+1 消费）。"""

    tokens: list[int]  # [K] CPU（forward 输入组装 + 验证）
    logprobs: torch.Tensor  # [K, V] fp32 GPU（验证的 draft 分布）
    p_top1: list[float]  # [K]（置信度校准输入）


@dataclass(slots=True)
class SampleLaunch:
    """一步采样的在途状态（ZOS：launch → 异步 D2H → 下一步 post_process 提交）。"""

    request_ids: list[str]
    pinned: torch.Tensor  # [R] int32 CPU（pinned；CUDA 事件守卫）
    event: object | None  # torch.cuda.Event（无 CUDA 时 None）
    sched_out: "SchedulerOutput"  # 提交时需要（持有以简化引擎侧簿记）
    #: MTP 验证行的结果（同步验证模式：sample 阶段已得到 CPU 列表；
    #: 多 token/seq）
    tokens_dict: dict[str, list[int]] | None = None

    def wait_tokens(self) -> dict[str, list[int]]:
        """唯一 CPU-GPU 同步点：等待采样 + D2H 完成 → request_id → tokens。"""
        out: dict[str, list[int]] = {}
        if self.tokens_dict:
            out.update(self.tokens_dict)
        if self.request_ids:
            if self.event is not None:
                self.event.synchronize()
            for rid, t in zip(self.request_ids, self.pinned.tolist()):
                out[rid] = [int(t)]
        return out


class ModelRunner:
    __slots__ = ("engine_cfg", "model", "cfg", "cache", "device", "dtype", "use_absorbed", "sampler")

    def __init__(
        self,
        engine_cfg: EngineConfig,
        model_dir: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_blocks: int | None = None,
        use_absorbed: bool = True,
        model: torch.nn.Module | None = None,
        model_cfg: ModelConfig | None = None,
        draft_model=None,
    ) -> None:
        self.engine_cfg = engine_cfg
        self.device = device
        self.dtype = dtype
        self.use_absorbed = use_absorbed
        self.sampler = Sampler()
        #: MTP：草案模型（DraftModel 协议）与上一轮草案状态
        self.draft_model = draft_model
        self._mtp_drafts: dict[str, "DraftInfo"] = {}
        self.calibrator = ConfidenceCalibrator()
        #: SO：matcher 句柄 + 本步失败集合（失败隔离）
        self._so_matchers: dict[str, "SOMatcher"] = {}
        self._so_failed: set[str] = set()
        if model is not None:
            assert model_cfg is not None
            self.model, self.cfg = model, model_cfg
        else:
            assert model_dir is not None
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
            if s.is_prefill:
                toks = [seq.token_at(p) for p in range(start, start + s.num_new_tokens)]
            else:
                # decode 行：首 token = 已提交末 token；后续 = 草案 token
                # （无草案 → 0 哨兵：验证走 fresh 拒绝，垃圾 KV 随回滚释放）
                first = seq.token_at(start)
                if s.num_new_tokens > 1:
                    draft = self._mtp_drafts.get(seq.request_id)
                    rest = list(draft.tokens) if draft else [0] * (s.num_new_tokens - 1)
                else:
                    rest = []
                toks = [first, *rest]
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

    # ------------------------------------------------------------------ SO

    def attach_so_matcher(self, request_id: str, matcher: SOMatcher) -> None:
        self._so_matchers[request_id] = matcher

    def detach_so_matcher(self, request_id: str) -> None:
        self._so_matchers.pop(request_id, None)
        self._so_failed.discard(request_id)

    @property
    def so_failed_requests(self) -> list[str]:
        return list(self._so_failed)

    def _fill_mask_row(self, matcher: SOMatcher, words: list[int]) -> None:
        """terminated → 全开（DeepInfer EOS 后全开语义）。"""
        if matcher.terminated:
            matcher.fill_ones(words)
        else:
            matcher.fill_mask_into(words)

    def _build_plain_so_masks(self, plain_sel: list) -> tuple[object, object] | None:
        """普通采样行的位掩码：行 0 = 全 1（无约束）；SO 行 1..。
        Returns: (so_bitmask GPU int32 [rows, words], row_map int32) 或 None。"""
        import torch as _torch

        so_idx: dict[str, int] = {}
        rows: list[list[int]] = [[-1] * self._so_words]  # 行 0：全 1
        for rid, _scalars, _pos, _golden in plain_sel:
            m = self._so_matchers.get(rid)
            if m is None or m.status == -1:
                continue
            row = [0] * self._so_words
            try:
                self._fill_mask_row(m, row)
            except Exception:  # 失败隔离
                self._so_failed.add(rid)
                m.status = -1
                continue
            so_idx[rid] = len(rows)
            rows.append(row)
        if not so_idx:
            return None
        row_map = _torch.tensor(
            [so_idx.get(rid, 0) for rid, _, _, _ in plain_sel],
            dtype=_torch.int32,
            device=self.device,
        )
        bitmask = _torch.tensor(rows, dtype=_torch.int32, device=self.device)
        return bitmask, row_map

    @property
    def _so_words(self) -> int:
        return (self.cfg.vocab_size + 31) // 32

    def accept_so_committed(self, request_id: str, tokens: list[int]) -> None:
        """post_process 时推进 matcher 真实状态（MTP/普通行统一入口）。"""
        m = self._so_matchers.get(request_id)
        if m is None or m.status == -1:
            return
        try:
            for t in tokens:
                m.accept_token(int(t))
            m.status = 2
        except Exception:
            self._so_failed.add(request_id)
            m.status = -1

    @staticmethod
    def decode_batch_meta(n: int) -> ForwardBatch:
        """纯 decode 批的采样元数据（launch_sample 只读 sample_flags /
        segment_offsets / segment_lens 三个字段；CUDA graph 路径使用）。"""
        return ForwardBatch(
            input_ids=None,  # type: ignore[arg-type]
            positions=None,  # type: ignore[arg-type]
            seq_ids=None,  # type: ignore[arg-type]
            segment_lens=[1] * n,
            block_tables=None,  # type: ignore[arg-type]
            sample_flags=[True] * n,
            segment_offsets=list(range(n)),
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
        sel_sched = []  # 与 sel 对齐的 ScheduledSeq（MTP 分流需要）
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
            sel_sched.append(s)
        if not sel:
            return None
        # ---- MTP 验证行（decode 行且 K>0）：同步验证（正确性优先路径；
        # fused Triton kernel 后置以恢复全异步）----
        mtp_tokens: dict[str, list[int]] = {}
        plain_sel = []
        plain_rows: list[int] = []
        K = self.engine_cfg.scheduler.nextn_draft_tokens
        for (rid, scalars, pos, golden), row_idx, s in zip(sel, rows_idx, sel_sched):
            if (not s.is_prefill) and K > 0:
                committed = self._verify_mtp_seq(logits, s, scalars.seq_seed)
                mtp_tokens[rid] = committed
            else:
                plain_sel.append((rid, scalars, pos, golden))
                plain_rows.append(row_idx)
        # ---- 草案发射（下一步 forward 输入）----
        if self.draft_model is not None:
            self._launch_drafts(sched_out, mtp_tokens)
        # ---- 普通行：确定性采样 kernel + 异步 D2H ----
        if plain_sel:
            rows = build_sampling_rows(plain_sel, self.device)
            logits_rows = logits[plain_rows].to(torch.float32).contiguous()
            so = self._build_plain_so_masks(plain_sel)
            if so is not None:
                tokens = self.sampler.sample_tensors(logits_rows, rows, so_bitmask=so[0], so_row_map=so[1])
            else:
                tokens = self.sampler.sample_tensors(logits_rows, rows)
            is_cuda = str(self.device).startswith("cuda")
            pinned = torch.empty(
                tokens.shape, dtype=torch.int32, device="cpu", pin_memory=is_cuda
            )
            pinned.copy_(tokens, non_blocking=is_cuda)
            event = None
            if is_cuda:
                event = torch.cuda.Event()
                event.record()
            return SampleLaunch(
                request_ids=[rid for rid, _, _, _ in plain_sel],
                pinned=pinned,
                event=event,
                sched_out=sched_out,
                tokens_dict=mtp_tokens or None,
            )
        # 全部为 MTP 行：结果已在 CPU
        return SampleLaunch(
            request_ids=[],
            pinned=torch.empty(0, dtype=torch.int32),
            event=None,
            sched_out=sched_out,
            tokens_dict=mtp_tokens or None,
        )

    # ------------------------------------------------------------------ MTP

    def _verify_mtp_seq(self, logits: torch.Tensor, s, seq_seed: int) -> list[int]:
        """单 seq 的 MTP 验证：logits 段 [1+K, V] + 上轮草案 → 提交 token。

        SO 集成（DeepInfer matcher.py:366-376 语义）：行 j 的掩码 = 试探性
        接受 d_0..d_{j-1} 后的状态；bonus 行 = 全接受态；填完回滚全部试探
        （真实状态由 post_process 的 committed 统一推进——见
        accept_so_committed）。掩码在 log_softmax 之前施加于 logits。
        """
        from cascade.sampling.mtp_verify import mtp_verify
        from cascade.sampling.rng import derive_uniform
        from cascade.model_executor.triton_ops.apply_so_bitmask import apply_token_bitmask
        from cascade.model_executor.triton_ops.common import CASE_SAMPLE, CASE_VERIFY

        seq = s.seq
        off = seq.num_tokens_computed - s.num_new_tokens  # 段首（本 runner 批内偏移）
        rows = logits[off : off + s.num_new_tokens].to(torch.float32)
        K_actual = s.num_new_tokens - 1
        draft = self._mtp_drafts.pop(seq.request_id, None)
        P = seq.num_tokens - 1  # 输入首行（已提交末 token）的绝对位置
        dev = logits.device
        if draft is not None and len(draft.tokens) == K_actual and K_actual > 0:
            d_tokens = torch.tensor(draft.tokens, dtype=torch.int64, device=dev)
            d_lp = draft.logprobs
            d_list = list(draft.tokens)
        else:
            d_tokens = torch.full((K_actual,), -1, dtype=torch.int64, device=dev)
            d_lp = torch.zeros(K_actual, logits.shape[-1], dtype=torch.float32, device=dev)
            d_list = None
        # ---- SO 多步试探掩码 ----
        matcher = self._so_matchers.get(seq.request_id)
        if matcher is not None and matcher.status != -1:
            try:
                words = self._so_words
                masks = [[0] * words for _ in range(s.num_new_tokens)]
                for j in range(K_actual):
                    self._fill_mask_row(matcher, masks[j])
                    matcher.accept_token(d_list[j] if d_list is not None else 0)
                if K_actual > 0:
                    self._fill_mask_row(matcher, masks[K_actual])  # bonus 行（全接受态）
                    matcher.rollback(K_actual)
                bitmask = torch.tensor(masks, dtype=torch.int32, device=dev)
                row_map = torch.arange(s.num_new_tokens, dtype=torch.int32, device=dev)
                apply_token_bitmask(rows, bitmask, row_map)  # log_softmax 之前
            except Exception:
                self._so_failed.add(seq.request_id)
                matcher.status = -1
        lp = torch.log_softmax(rows, dim=-1)
        accept_u = torch.tensor(
            [derive_uniform(seq_seed, P + j + 1, CASE_VERIFY) for j in range(K_actual)],
            dtype=torch.float32,
        )
        sample_u = torch.tensor(
            [derive_uniform(seq_seed, P + j + 1, CASE_SAMPLE) for j in range(K_actual + 1)],
            dtype=torch.float32,
        )
        res = mtp_verify(lp, d_lp, d_tokens, accept_u, sample_u)
        # 置信度校准闭环（draft p_top1 ↔ 真实接受概率估计）
        if draft is not None:
            self.calibrator.update_batch(draft.p_top1[: len(res.accept_probs)], res.accept_probs)
        return res.committed

    def _launch_drafts(self, sched_out, mtp_tokens: dict[str, list[int]]) -> None:
        """为下一步发射草案：输入 = 本步提交的末 token（同步验证模式下
        已在 CPU）。prefill 收尾的 seq 首个 decode 步无草案（fresh 行）。"""
        new_drafts: dict[str, DraftInfo] = {}
        K = self.engine_cfg.scheduler.nextn_draft_tokens
        dev = self.device
        for s in sched_out.scheduled:
            if s.is_prefill:
                continue
            committed = mtp_tokens.get(s.seq.request_id)
            if not committed:
                continue  # 验证未产出（不应发生）或本轮无该 seq 结果
            last_token = committed[-1]
            start_pos = s.seq.num_tokens - 1 + len(committed)  # 末 token 绝对位置
            tokens, logprobs, p_top1 = self.draft_model.draft_tokens(
                last_hidden=None,
                last_token=torch.tensor([last_token], dtype=torch.int64, device=dev),
                start_positions=torch.tensor([start_pos], dtype=torch.int64, device=dev),
                k=K,
            )
            new_drafts[s.seq.request_id] = DraftInfo(
                tokens=[int(t) for t in tokens[0].tolist()],
                logprobs=logprobs[0],
                p_top1=[float(x) for x in p_top1[0].tolist()],
            )
        self._mtp_drafts = new_drafts  # 旧条目整体淘汰（含已结束/被抢占 seq）


__all__ = ["ModelRunner", "ForwardBatch", "SampleLaunch"]

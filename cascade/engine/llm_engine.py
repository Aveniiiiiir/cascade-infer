"""同步引擎：调度器 × ModelRunner 的闭环。

执行序（审查 #2 修复，v2）——**同步序**：schedule → prepare → forward →
sample → wait → commit，一步内闭环。

为何不是 DeepInfer 的 ZOS 三段式（Forward(N) → PostProcess(N−1) →
Sample(N)）：DeepInfer 的滞后一步依赖「上一步采样 token 全程留在 GPU」——
worker.py `_correct_seqlens_qk` 把 `prev.sampler_output.sampled_token_ids_gpu`
传给 MetadataZOS，由 `gather_input_ids_zos` 在**设备侧**组装本步输入 token，
CPU 从不在 forward 前读取它。Cascade 的 `prepare_batch` 在 CPU 组 token
（`seq.token_at`），一步滞后会在第二个 decode 步读取未提交 token（已用纯
调度器复现 IndexError）。ZOS 的 GPU 侧喂数属于后续 Phase 的重构项，本
实现先恢复正确闭环——调度与执行的重叠收益在 Cascade 的 Python 组批开销
下本也有限。

MTP 的乐观推进/提交回滚协议不变（scheduler._commit_optimistic_advance）。
"""

from __future__ import annotations

from cascade.common.logger import get_logger
from cascade.config import EngineConfig
from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.scheduler import Scheduler
from cascade.core.sequence import RequestOutput
from cascade.sampling_params import SamplingParams
from cascade.structured_output.builder import SOMatchersBuilder
from cascade.worker.model_runner import ModelRunner

_log = get_logger("engine.llm_engine")


class LLMEngine:
    # 审查 #1：__slots__ 必须覆盖 __init__ 赋值的全部属性（漏声明 = 构造即
    # AttributeError）。曾有 _so_*/graph_runner 漏写导致引擎从未成功构造。
    __slots__ = (
        "cfg", "scheduler", "runner", "block_mgr", "_request_counter",
        "_so_builder", "_so_staging", "_so_failed_outputs", "graph_runner",
    )

    def __init__(
        self,
        engine_cfg: EngineConfig,
        model_dir: str | None = None,
        device: str = "cuda",
        dtype=None,
        num_blocks: int | None = None,
        draft_model=None,  # DraftModel 协议（MTP；None = 关闭）
        model=None,  # 预构建模型（引擎级测试免 checkpoint）
        model_cfg=None,
    ) -> None:
        import torch

        from cascade.common.env import envs

        if dtype is None:
            dtype = torch.bfloat16
        if draft_model is not None and engine_cfg.scheduler.nextn_draft_tokens == 0:
            # 默认验证长度取 env（DeepInfer NEXTN_DECODE 默认 5；Cascade 默认 3）。
            # 注意：这里变异调用方传入的 engine_cfg（共享对象）——引擎构造后
            # cfg 即引擎所有，调用方不应复用（审查 #23 注记）。
            engine_cfg.scheduler.nextn_draft_tokens = envs.NEXTN_DECODE
            engine_cfg.validate()  # 重算 runtime_max_model_len（K 计入位置余量）
        self.cfg = engine_cfg
        self.runner = ModelRunner(
            engine_cfg, model_dir, device=device, dtype=dtype, num_blocks=num_blocks,
            model=model, model_cfg=model_cfg, draft_model=draft_model,
        )
        num_blocks = self.runner.cache.num_blocks
        self.block_mgr = BlockManager(num_blocks)
        prefix_cache = (
            PrefixCache(self.block_mgr, engine_cfg.cache.block_size)
            if engine_cfg.scheduler.enable_prefix_caching
            else None
        )
        self.scheduler = Scheduler(engine_cfg, self.block_mgr, prefix_cache)
        self._request_counter = 0
        #: SO：异步编译 builder + staging（编完才入调度）+ 失败输出缓冲
        self._so_builder = SOMatchersBuilder()
        self._so_staging: dict[str, tuple[list[int], SamplingParams]] = {}
        self._so_failed_outputs: list[RequestOutput] = []
        # rope 缓冲一次性预热到 runtime_max_model_len：此后 eager 与图捕获
        # 路径全部走 rope_ensure=False（每层免 .item() 同步；捕获后重分配
        # 会造成已捕获指针 use-after-free —— 审查 #5/#20）。
        _prewarm_rope(self.runner.model, engine_cfg.runtime_max_model_len)
        self.graph_runner = self._init_graph_runner(engine_cfg, device, dtype)
        _log.info(
            "LLMEngine ready (同步序): model=%s blocks=%d prefix_cache=%s cudagraph=%s",
            model_dir,
            num_blocks,
            prefix_cache is not None,
            self.graph_runner is not None,
        )

    def _init_graph_runner(self, engine_cfg, device, dtype):
        import torch

        from cascade.common.env import envs

        if not envs.CUDAGRAPH:
            return None
        if not str(device).startswith("cuda") or dtype not in (torch.bfloat16, torch.float16):
            return None
        if not self.runner.cache or engine_cfg.scheduler.max_num_seqs < 1:
            return None
        from cascade.worker.cuda_graph_runner import CudaGraphRunner

        # 预留 scratch 块（分配即持有；图 padding 行的 KV 写入目标）
        scratch = self.block_mgr.allocate()
        try:
            return CudaGraphRunner(
                self.runner.model, self.runner.cache, engine_cfg, device,
                max_batch_size=engine_cfg.scheduler.max_num_seqs,
                scratch_block=scratch,
            )
        except Exception:  # 图捕获失败（显存/驱动限制）→ 优雅降级 eager
            self.block_mgr.release(scratch)
            _log.warning("CUDA graph capture failed → fallback to eager", exc_info=True)
            return None

    @property
    def model_cfg(self):
        return self.runner.cfg

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        has_so: bool = False,
    ) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if request_id is None:
            request_id = f"req-{self._request_counter}"
            self._request_counter += 1
        if sampling_params.so_args:
            # SO：异步编译，编完才入调度（DeepInfer 编译不阻塞调度热路径）
            self._so_staging[request_id] = (prompt_token_ids, sampling_params)
            self._so_builder.submit(
                request_id, sampling_params.so_args, self.cfg.model.vocab_size
            )
            return request_id
        self.scheduler.add_request(request_id, prompt_token_ids, sampling_params, has_so=has_so)
        return request_id

    def abort(self, request_id: str) -> bool:
        """审查 #11：serving 层（AsyncEngine.abort）依赖此入口——此前缺失，
        一调用即 AttributeError。"""
        return self.scheduler.abort(request_id)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def step(self) -> list[RequestOutput]:
        """同步序引擎步：schedule → prepare/forward → sample → commit。

        见类 docstring：为何不是 DeepInfer 的一步滞后 ZOS（token 的 GPU 侧
        喂入是前置条件，Cascade 尚无该设施）。
        """
        self._pump_so()
        self._drain_so_failures()
        outputs: list[RequestOutput] = list(self._so_failed_outputs)
        self._so_failed_outputs.clear()
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            # 空步也要提交（弹出 pending 里的空 SchedulerOutput）——否则
            # 空步累积 pending，下一步 schedule 的深度断言锁死引擎
            outputs.extend(self.scheduler.update_from_output({}))
            return outputs
        graph_path = self._try_graph_forward(sched_out)
        if graph_path is not None:
            logits, batch = graph_path
        else:
            batch = self.runner.prepare_batch(sched_out)
            logits = self.runner.forward(batch, rope_ensure=False)  # 已预热
        try:
            launch = self.runner.launch_sample(logits, sched_out, batch)
            sampled = launch.wait_tokens() if launch is not None else {}
        except Exception:
            # 采样/验证异常：必须弹出本步 pending（否则下一步 schedule 的
            # 深度断言会把引擎锁死）。判死全部在途请求后重抛——引擎线程
            # 的 fail_all_inflight 兜底在服务层，引擎层先复位 pending。
            self.scheduler.fail_all_inflight()
            raise
        outputs.extend(self.scheduler.update_from_output(sampled))
        # SO：统一推进 matcher 真实状态（MTP/普通行）；结束的 seq 卸载
        for ro in outputs:
            if ro.new_token_ids:
                self.runner.accept_so_committed(ro.request_id, ro.new_token_ids)
            if ro.finish_reason is not None:
                self.runner.detach_so_matcher(ro.request_id)
        return outputs

    def _try_graph_forward(self, sched_out):
        """纯 decode 批且档位可容纳 → CUDA graph 回放。

        Returns: (logits [n, vocab] bf16, decode 元数据 batch) 或 None。
        仅接受 decode 行（审查补充：1-token 的 mid-chunk prefill 行曾能
        混入 → sample_flags 失配 → 提交断言崩溃）。
        """
        import torch

        if self.graph_runner is None:
            return None
        scheduled = sched_out.scheduled
        n = len(scheduled)
        if n == 0 or any(s.is_prefill or s.num_new_tokens != 1 for s in scheduled):
            return None
        if not self.graph_runner.can_run(n):
            return None
        tokens, positions, block_rows = [], [], []
        for s in scheduled:
            seq = s.seq
            pos = seq.num_tokens_computed - 1
            tokens.append(seq.token_at(pos))
            positions.append(pos)
            block_rows.append(seq.block_table)
        hidden = self.graph_runner.replay(tokens, positions, block_rows)  # [T, H]
        logits = self.runner.model.lm_head(hidden[:n])  # 图外一次 GEMM
        # decode 元数据（launch_sample 只读 sample_flags / segment_offsets /
        # segment_lens 三个字段；segment_offsets 即批内行号）
        batch = self.runner.decode_batch_meta(n)
        return logits, batch

    def _pump_so(self) -> None:
        """收割 SO 编译结果：成功的入调度 + 挂 matcher；失败的直接判死
        （staged 请求从未入队 → 合成 FINISHED_SO_FAILED 输出）。"""
        if not self._so_staging and self._so_builder.num_pending == 0:
            return
        for result in self._so_builder.pump():
            staged = self._so_staging.pop(result.request_id, None)
            if staged is None:
                continue  # 已被 abort 等移除
            prompt, sp = staged
            if result.matcher is None:
                self._so_failed_outputs.append(
                    RequestOutput(
                        request_id=result.request_id,
                        new_token_ids=[],
                        output_token_ids=[],
                        finish_reason="so_failed",
                    )
                )
                continue
            self.runner.attach_so_matcher(result.request_id, result.matcher)
            try:
                self.scheduler.add_request(result.request_id, prompt, sp, has_so=True)
            except Exception:
                # staged prompt 非法（空/超长）：判死该请求而非杀死引擎
                # （审查 #4 同源问题的引擎侧兜底）
                self.runner.detach_so_matcher(result.request_id)
                self._so_failed_outputs.append(
                    RequestOutput(
                        request_id=result.request_id,
                        new_token_ids=[],
                        output_token_ids=[],
                        finish_reason="error",
                    )
                )

    def _drain_so_failures(self) -> None:
        """在途 SO 失败（掩码/接受异常）→ 判死本请求，引擎继续。"""
        for rid in self.runner.so_failed_requests:
            self.runner.detach_so_matcher(rid)
            self.scheduler.fail_so(rid)

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> dict[str, list[int]]:
        """离线批生成：所有请求跑完 → request_id → 输出 token 列表。

        sampling_params 可为单个（共享）或逐请求列表（批不变测试需要
        每请求独立 seq_seed）。
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompt_token_ids)
        else:
            assert len(sampling_params) == len(prompt_token_ids)
            params_list = list(sampling_params)
        ids = [self.add_request(p, sp) for p, sp in zip(prompt_token_ids, params_list)]
        results: dict[str, list[int]] = {i: [] for i in ids}
        finish: dict[str, str] = {}
        while self.scheduler.has_unfinished_requests or self._so_staging:
            for ro in self.step():
                results[ro.request_id].extend(ro.new_token_ids)
                if ro.finish_reason is not None:
                    finish[ro.request_id] = ro.finish_reason
        # 同步序无一步滞后；但判死请求的最终回执可能滞留在 _pending_req_outputs
        # （最后一步之后没有再进 update 路径）——显式冲刷（审查 #2 配套）。
        for ro in self.scheduler.flush_final_outputs():
            results[ro.request_id].extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish[ro.request_id] = ro.finish_reason
        assert not self.scheduler.has_unfinished_requests
        return results


def _prewarm_rope(model, runtime_max_model_len: int) -> None:
    """把模型的 rope cos/sin 缓冲一次性扩容到位（eager 与图捕获共用）。

    位置上界：批内最大位置 = computed−1 ≤ num_tokens−2+num_new ≤
    max_model_len+K−1 < runtime_max_model_len（EngineConfig.validate 的
    K 计入保证）。自定义模型无 rope 属性时跳过（测试桩）。"""
    rope = getattr(getattr(model, "model", None), "rope", None)
    if rope is not None and hasattr(rope, "_ensure"):
        rope._ensure(runtime_max_model_len)


__all__ = ["LLMEngine"]

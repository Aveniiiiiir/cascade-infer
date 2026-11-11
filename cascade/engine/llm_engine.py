"""同步引擎：调度器 × ModelRunner 的闭环（Phase 2）。

Phase 4 将把 step() 重构为 ZOS 三段式（forward → post_process(prev) →
sample），届时本类的职责不变、内部流水线重组 —— 接口按此预留。
"""

from __future__ import annotations

from cascade.common.logger import get_logger
from cascade.config import EngineConfig
from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.scheduler import Scheduler
from cascade.core.sequence import RequestOutput
from cascade.sampling_params import SamplingParams
from cascade.worker.model_runner import ModelRunner

_log = get_logger("engine.llm_engine")


class LLMEngine:
    __slots__ = ("cfg", "scheduler", "runner", "block_mgr", "_request_counter", "_prev_sample")

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
            # 默认验证长度取 env（DeepInfer NEXTN_DECODE 默认 5；Cascade 默认 3）
            engine_cfg.scheduler.nextn_draft_tokens = envs.CASCADE_NEXTN_DECODE
            engine_cfg.runtime_max_model_len = (
                engine_cfg.model.max_model_len
                + 2 * engine_cfg.model.num_nextn_predict_layers * 0  # K 行已由 1+K 计入预算
                + 1
            )
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
        self._prev_sample = None  # 上一步的 SampleLaunch（ZOS 在途状态）
        self.graph_runner = self._init_graph_runner(engine_cfg, device, dtype)
        _log.info(
            "LLMEngine ready (ZOS 三段式): model=%s blocks=%d prefix_cache=%s cudagraph=%s",
            model_dir,
            num_blocks,
            prefix_cache is not None,
            self.graph_runner is not None,
        )

    def _init_graph_runner(self, engine_cfg, device, dtype):
        import torch

        from cascade.common.env import envs

        if not envs.CASCADE_CUDAGRAPH:
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
        self.scheduler.add_request(request_id, prompt_token_ids, sampling_params, has_so=has_so)
        return request_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def step(self) -> list[RequestOutput]:
        """ZOS 三段式（对应 DeepInfer worker.py:89 的 Forward→PostProcess(prev)→Sample）：

        1. schedule(N) + prepare + **launch** forward(N)（CPU 立即返回；
           纯 decode 批走 CUDA graph 回放）
        2. post_process(N−1)：唯一同步点 = 上一步采样事件 → 提交调度器
        3. launch sample(N) + 异步 D2H（留给下一步的 post_process）

        GPU 跑 forward(N)→sample(N) 的同时，CPU 并发跑 schedule(N+1) 前的
        提交与产出逻辑 —— 调度与执行重叠，无空泡。
        """
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            # 无新工作：冲刷上一步在途采样后返回
            return self._post_process_prev()
        graph_path = self._try_graph_forward(sched_out)
        if graph_path is not None:
            logits, batch = graph_path
        else:
            batch = self.runner.prepare_batch(sched_out)
            logits = self.runner.forward(batch)  # 只 launch
        outputs = self._post_process_prev()  # 与 forward(N) GPU 执行重叠
        self._prev_sample = self.runner.launch_sample(logits, sched_out, batch)
        return outputs

    def _try_graph_forward(self, sched_out):
        """纯 decode 批且档位可容纳 → CUDA graph 回放。

        Returns: (logits [n, vocab] bf16, decode 元数据 batch) 或 None。
        """
        import torch

        if self.graph_runner is None:
            return None
        scheduled = sched_out.scheduled
        n = len(scheduled)
        if n == 0 or any(s.num_new_tokens != 1 for s in scheduled):
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
        # decode 元数据（launch_sample 只用这三个字段）
        batch = self.runner.decode_batch_meta(n)
        return logits, batch

    def _post_process_prev(self) -> list[RequestOutput]:
        if self._prev_sample is None:
            return []
        launch = self._prev_sample
        self._prev_sample = None
        sampled = launch.wait_tokens()  # 唯一 CPU-GPU 同步点
        return self.scheduler.update_from_output(sampled)

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
        while self.scheduler.has_unfinished_requests:
            for ro in self.step():
                results[ro.request_id].extend(ro.new_token_ids)
                if ro.finish_reason is not None:
                    finish[ro.request_id] = ro.finish_reason
        # 冲刷最后一个在途步骤（ZOS 尾部：1 步 lag 的提交 + 残留 pending）
        for ro in self._post_process_prev():
            results[ro.request_id].extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish[ro.request_id] = ro.finish_reason
        assert not self.scheduler.has_unfinished_requests
        return results


__all__ = ["LLMEngine"]

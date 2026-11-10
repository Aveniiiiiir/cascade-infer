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
    __slots__ = ("cfg", "scheduler", "runner", "block_mgr", "_request_counter")

    def __init__(
        self,
        engine_cfg: EngineConfig,
        model_dir: str,
        device: str = "cuda",
        dtype=None,
        num_blocks: int | None = None,
    ) -> None:
        import torch

        if dtype is None:
            dtype = torch.bfloat16
        self.cfg = engine_cfg
        self.runner = ModelRunner(engine_cfg, model_dir, device=device, dtype=dtype, num_blocks=num_blocks)
        num_blocks = self.runner.cache.num_blocks
        self.block_mgr = BlockManager(num_blocks)
        prefix_cache = (
            PrefixCache(self.block_mgr, engine_cfg.cache.block_size)
            if engine_cfg.scheduler.enable_prefix_caching
            else None
        )
        self.scheduler = Scheduler(engine_cfg, self.block_mgr, prefix_cache)
        self._request_counter = 0
        _log.info(
            "LLMEngine ready: model=%s blocks=%d prefix_cache=%s",
            model_dir,
            num_blocks,
            prefix_cache is not None,
        )

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
        """一步：调度 → 准备 → 前向 → 贪心采样 → 提交（Phase 3 换确定性采样器）。"""
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            return []
        batch = self.runner.prepare_batch(sched_out)
        logits = self.runner.forward(batch)
        sampled = self.runner.sample_greedy_rows(logits, sched_out, batch)
        return self.scheduler.update_from_output(sampled)

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, list[int]]:
        """离线批生成：所有请求跑完 → request_id → 输出 token 列表。"""
        ids = [self.add_request(p, sampling_params) for p in prompt_token_ids]
        results: dict[str, list[int]] = {i: [] for i in ids}
        finish: dict[str, str] = {}
        while self.scheduler.has_unfinished_requests:
            for ro in self.step():
                results[ro.request_id].extend(ro.new_token_ids)
                if ro.finish_reason is not None:
                    finish[ro.request_id] = ro.finish_reason
        return results


__all__ = ["LLMEngine"]

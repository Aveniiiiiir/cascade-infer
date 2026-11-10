"""Sequence / request state machine (CPU side, no torch dependency).

Mirrors DeepInfer's `csrc/engine/sequence.hpp` at the semantic level: a
`Sequence` owns its token history, KV block table, and a POD snapshot of the
sampling scalars so the scheduler hot path never touches the original
`SamplingParams` object. Block id 0 is reserved as the padding/empty block
(never allocated) so GPU kernels can use table row 0 as a no-op slot later.
"""

from __future__ import annotations

from enum import Enum

from cascade.sampling_params import SamplingScalars


class SequenceStatus(Enum):
    WAITING = "waiting"  # prompt 未算完（含 chunked 中间态）
    RUNNING = "running"  # prefill 完成，可参与 decode
    FINISHED_STOPPED = "finished_stopped"  # 命中 stop token
    FINISHED_LENGTH_CAPPED = "finished_length_capped"  # 达到 max_tokens / max_model_len
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_PREEMPTED = "finished_preempted"  # stop 模式抢占（直接结束）
    FINISHED_SO_FAILED = "finished_so_failed"  # 结构化输出失败隔离（Phase 5）

    @property
    def is_finished(self) -> bool:
        return self.name.startswith("FINISHED")


_FINISH_REASON = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_PREEMPTED: "preempt",
    SequenceStatus.FINISHED_SO_FAILED: "so_failed",
}


class Sequence:
    """One request's scheduler-side state. `__slots__` POD style."""

    __slots__ = (
        "request_id",
        "prompt_token_ids",
        "output_token_ids",
        "sampling",
        "status",
        "finish_reason",
        "num_tokens_computed",
        "block_table",
        "num_cached_tokens",
        "has_so",
        "_num_prompt_tokens",
    )

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling: SamplingScalars,
        has_so: bool = False,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError(f"request {request_id}: empty prompt")
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self._num_prompt_tokens = len(self.prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.sampling = sampling
        self.status = SequenceStatus.WAITING
        self.finish_reason: str | None = None
        #: KV 已计算到的 token 位置（prompt+output 前缀长度）
        self.num_tokens_computed = 0
        #: KV block 表（id 0 保留，永不出现）
        self.block_table: list[int] = []
        #: 前缀缓存命中长度（block 对齐）
        self.num_cached_tokens = 0
        #: SO 请求不可 RECOMPUTE 抢占（DeepInfer 语义）
        self.has_so = has_so

    # ------------------------------------------------------------------ lengths

    @property
    def num_tokens(self) -> int:
        return self._num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return self._num_prompt_tokens

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def token_at(self, position: int) -> int:
        """token 前缀(prompt‖output)中 position 处的 token id。"""
        if position < self._num_prompt_tokens:
            return self.prompt_token_ids[position]
        return self.output_token_ids[position - self._num_prompt_tokens]

    @property
    def last_token_id(self) -> int:
        return self.token_at(self.num_tokens - 1)

    # ------------------------------------------------------------------ mutate

    def append_output_tokens(self, token_ids: list[int]) -> None:
        assert token_ids, "append_output_tokens: empty list"
        assert self.status in (SequenceStatus.WAITING, SequenceStatus.RUNNING)
        self.output_token_ids.extend(token_ids)

    def finish(self, status: SequenceStatus) -> None:
        assert status.is_finished, f"finish() with non-terminal status {status}"
        assert not self.is_finished, f"finish() on already-finished {self.request_id}"
        self.status = status
        self.finish_reason = _FINISH_REASON.get(status, "unknown")

    def num_blocks_for(self, num_tokens: int, block_size: int) -> int:
        """容纳 num_tokens 个 KV 所需的 block 数（向上取整）。"""
        return (num_tokens + block_size - 1) // block_size

    # ------------------------------------------------------------------ debug

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Sequence({self.request_id}, {self.status.value}, "
            f"prompt={self._num_prompt_tokens}, out={len(self.output_token_ids)}, "
            f"computed={self.num_tokens_computed}, blocks={len(self.block_table)})"
        )


class RequestOutput:
    """Delta output for one step of one request (DeepInfer RequestOutput 语义:
    池化对象下一步被回收，消费方需要持久化时必须拷贝 —— CPU 版为不可变元组式)。"""

    __slots__ = ("request_id", "new_token_ids", "output_token_ids", "finish_reason", "num_cached_tokens")

    def __init__(
        self,
        request_id: str,
        new_token_ids: list[int],
        output_token_ids: list[int],
        finish_reason: str | None,
        num_cached_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.new_token_ids = list(new_token_ids)
        self.output_token_ids = list(output_token_ids)
        self.finish_reason = finish_reason
        self.num_cached_tokens = num_cached_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RequestOutput({self.request_id}, new={len(self.new_token_ids)}, "
            f"total={len(self.output_token_ids)}, finish={self.finish_reason})"
        )


__all__ = ["Sequence", "SequenceStatus", "RequestOutput"]

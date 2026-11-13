"""Sampling parameters with a POD snapshot for the scheduler hot path.

Follows DeepInfer's `sampling_params.py` semantics:
- `seq_seed` (uint64) drives the deterministic RNG stream; combined with the
  absolute token position it makes sampling independent of batch composition.
- Greedy degeneration: temperature below 1e-5 forces `temperature=eps,
  top_p=1, top_k=-1` so greedy and sampled paths share one kernel.
- `so_args` carries structured-output constraints (Phase 5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1
_GREEDY_EPS = 1e-5
#: 独立的 draft 随机流偏移（DeepInfer fused_sample.py:15 `_MTP_SEED_OFFSET`）
MTP_SEED_OFFSET = 1000000007


@dataclass(slots=True)
class SamplingScalars:
    """POD snapshot captured once at enqueue time (DeepInfer SamplingScalars).

    The scheduler hot path reads only this; the original SamplingParams object
    is never touched after add_request.
    """

    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seq_seed: int
    ignore_eos: bool
    stop_token_ids: tuple[int, ...]
    #: 多 token stop 序列（后缀匹配；调度器在提交点检测并截断——见
    #: Scheduler._truncate_new_tokens 的语义注记）
    stop_token_seqs: tuple[tuple[int, ...], ...] = ()
    #: teacher-forcing 输出（测试/回放：第 output_len 个强制输出）
    golden_token_ids: tuple[int, ...] = ()

    def is_stop_token(self, token_id: int) -> bool:
        return not self.ignore_eos and token_id in self.stop_token_ids

    def golden_at(self, output_len: int) -> int:
        return self.golden_token_ids[output_len] if output_len < len(self.golden_token_ids) else -1


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    seq_seed: int | None = None
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()
    #: 多 token stop 序列（token id 元组；serving 层把 stop 字符串 tokenize 后传入）
    stop_token_seqs: tuple[tuple[int, ...], ...] | list[list[int]] | None = None
    #: 结构化输出约束: {"schema_type": "regex"|"json"|"ebnf", "schema": ..., ...}
    so_args: dict | None = None
    #: teacher-forcing 输出（测试/回放用，DeepInfer golden_token_ids）
    golden_token_ids: list[int] | None = None
    _extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.stop_token_ids, list):
            self.stop_token_ids = tuple(self.stop_token_ids)
        if self.stop_token_seqs:
            # 规范化为 tuple[tuple[int, ...], ...]；空序列剔除（永不匹配）
            self.stop_token_seqs = tuple(
                tuple(int(t) for t in seq) for seq in self.stop_token_seqs if len(seq)
            )
        else:
            self.stop_token_seqs = ()
        # 贪心退化（DeepInfer sampling_params.py:76-79）
        if self.temperature < _GREEDY_EPS:
            self.temperature = _GREEDY_EPS
            self.top_p = 1.0
            self.top_k = -1
        if self.seq_seed is None:
            self.seq_seed = random.getrandbits(64)
        if not (0 <= self.seq_seed <= _UINT64_MAX):
            raise ValueError(f"seq_seed must be in [0, 2^64), got {self.seq_seed}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        # 审查 #9：top_k 只允许 -1（不截断）或 [1, int32 max]。≥ vocab 的值
        # 由 Sampler 侧 clamp 为"不截断"（kth=最小值），但 0 / < -1 直接拒绝。
        if self.top_k < -1 or self.top_k == 0 or self.top_k > _INT32_MAX:
            raise ValueError(
                f"top_k must be -1 (disable) or in [1, 2^31-1], got {self.top_k}"
            )

    def snapshot(self) -> SamplingScalars:
        golden = tuple(self.golden_token_ids) if self.golden_token_ids else ()
        return SamplingScalars(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            seq_seed=self.seq_seed,
            ignore_eos=self.ignore_eos,
            stop_token_ids=self.stop_token_ids,
            stop_token_seqs=self.stop_token_seqs,
            golden_token_ids=golden,
        )

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= _GREEDY_EPS

    def is_stop_token(self, token_id: int) -> bool:
        return not self.ignore_eos and token_id in self.stop_token_ids


__all__ = ["SamplingParams", "SamplingScalars", "MTP_SEED_OFFSET"]

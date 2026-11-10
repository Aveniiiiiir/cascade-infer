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

    def is_stop_token(self, token_id: int) -> bool:
        return not self.ignore_eos and token_id in self.stop_token_ids


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    seq_seed: int | None = None
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()
    #: 结构化输出约束: {"schema_type": "regex"|"json"|"ebnf", "schema": ..., ...}
    so_args: dict | None = None
    #: teacher-forcing 输出（测试/回放用，DeepInfer golden_token_ids）
    golden_token_ids: list[int] | None = None
    _extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.stop_token_ids, list):
            self.stop_token_ids = tuple(self.stop_token_ids)
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

    def snapshot(self) -> SamplingScalars:
        return SamplingScalars(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            seq_seed=self.seq_seed,
            ignore_eos=self.ignore_eos,
            stop_token_ids=self.stop_token_ids,
        )

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= _GREEDY_EPS

    def is_stop_token(self, token_id: int) -> bool:
        return not self.ignore_eos and token_id in self.stop_token_ids


__all__ = ["SamplingParams", "SamplingScalars", "MTP_SEED_OFFSET"]

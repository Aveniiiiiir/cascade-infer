"""Mock model executor for CPU closed-loop scheduler tests.

Deterministic "model": next token = f(seq_seed, absolute token position),
with an EOS emitted when the request's target output length is reached.
Because the function depends only on (seed, position), generated outputs are
invariant to batching, chunking, preemption, and pool size — exactly the
property the scheduler must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cascade.core.scheduler import Scheduler, SchedulerOutput
from cascade.sampling.rng import derive_uniform

#: 与 DeepSeek vocab 无关的测试用小词表
MOCK_VOCAB_SIZE = 4096
MOCK_EOS_TOKEN_ID = 3


@dataclass
class MockRequest:
    request_id: str
    prompt: list[int]
    seq_seed: int
    target_output_len: int  # 到达后出 EOS
    max_tokens: int = 10_000


@dataclass
class MockResult:
    outputs: dict[str, list[int]] = field(default_factory=dict)
    finish_reasons: dict[str, str] = field(default_factory=dict)
    cached_tokens: dict[str, int] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    num_steps: int = 0


class MockModel:
    def __init__(self, requests: list[MockRequest], vocab_size: int = MOCK_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = MOCK_EOS_TOKEN_ID
        self.targets = {r.request_id: r.target_output_len for r in requests}

    def step(self, out: SchedulerOutput) -> dict[str, list[int]]:
        sampled: dict[str, list[int]] = {}
        for scheduled in out.scheduled:
            if scheduled.is_prefill and not scheduled.completes_prefill:
                continue  # 中间 chunk 不产出 token
            seq = scheduled.seq
            position = seq.num_tokens  # 新 token 的绝对位置（append 前）
            if seq.num_output_tokens + 1 >= self.targets[seq.request_id]:
                token = self.eos_token_id
            else:
                token = int(derive_uniform(seq.sampling.seq_seed, position) * self.vocab_size)
                if token == self.eos_token_id:  # 避免随机撞上 EOS
                    token = (token + 1) % self.vocab_size
            sampled[seq.request_id] = [token]
        return sampled


def expected_tokens(req: MockRequest, mock: MockModel) -> list[int]:
    """直接用 mock 公式推导期望输出（不经调度器）——闭环测试的 oracle。"""
    tokens: list[int] = []
    for k in range(req.target_output_len):
        if k + 1 >= req.target_output_len:
            tokens.append(mock.eos_token_id)
            break
        position = len(req.prompt) + k
        token = int(derive_uniform(req.seq_seed, position) * mock.vocab_size)
        if token == mock.eos_token_id:
            token = (token + 1) % mock.vocab_size
        tokens.append(token)
    return tokens

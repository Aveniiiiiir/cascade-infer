"""Closed-loop engine harness: drives the scheduler with the mock model and
checks structural invariants after every step (the CPU-side acceptance gate).
"""

from __future__ import annotations

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.scheduler import Scheduler
from cascade.core.sequence import SequenceStatus
from cascade.sampling_params import SamplingParams
from tests.cpu.mock_model import MOCK_EOS_TOKEN_ID, MockModel, MockRequest, MockResult


def make_config(
    *,
    block_size: int = 64,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = True,
    max_model_len: int = 8192,
) -> EngineConfig:
    return EngineConfig(
        model=ModelConfig(max_model_len=max_model_len, vocab_size=4096, num_layers=1),
        cache=CacheConfig(block_size=block_size),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
        ),
    )


def _check_invariants(sched: Scheduler) -> None:
    """每步 update 后必须成立的硬不变式。"""
    sched.block_mgr.debug_check_consistency()
    if sched.prefix_cache is not None:
        sched.prefix_cache.debug_check_consistency()

    # 1. 每个 block 的引用计数 == 活跃 seq block 表中的出现次数
    occurrences: dict[int, int] = {}
    for seq in list(sched.waiting) + sched.running:
        for b in seq.block_table:
            occurrences[b] = occurrences.get(b, 0) + 1
    for b in range(1, sched.block_mgr.num_blocks + 1):
        expected = occurrences.get(b, 0)
        actual = sched.block_mgr.ref_count(b)
        assert actual == expected, f"block {b}: ref={actual}, table occurrences={expected}"

    # 2. block 表长度 == ceil(computed / block_size)（恰好分配，不多不少）
    for seq in list(sched.waiting) + sched.running:
        expected_blocks = (seq.num_tokens_computed + sched.block_size - 1) // sched.block_size
        assert len(seq.block_table) == expected_blocks, (
            f"{seq.request_id}: {len(seq.block_table)} blocks vs computed "
            f"{seq.num_tokens_computed} (expect {expected_blocks})"
        )

    # 3. RUNNING seq: computed == num_tokens - 1（末 token 是下一步的输入）
    for seq in sched.running:
        assert seq.num_tokens_computed == seq.num_tokens - 1, (
            f"running {seq.request_id}: computed={seq.num_tokens_computed} "
            f"num_tokens={seq.num_tokens}"
        )
    # 4. WAITING seq: computed < num_tokens（chunked/未开始/待重算）
    for seq in sched.waiting:
        assert seq.num_tokens_computed < seq.num_tokens


def run_engine(
    requests: list[MockRequest],
    cfg: EngineConfig,
    num_blocks: int,
    *,
    enable_prefix_caching: bool | None = None,
    max_steps: int = 100_000,
) -> MockResult:
    """闭环跑完所有请求，逐步校验不变式。"""
    if enable_prefix_caching is not None:
        cfg.scheduler.enable_prefix_caching = enable_prefix_caching
    block_mgr = BlockManager(num_blocks)
    prefix_cache = PrefixCache(block_mgr, cfg.cache.block_size) if cfg.scheduler.enable_prefix_caching else None
    sched = Scheduler(cfg, block_mgr, prefix_cache)
    mock = MockModel(requests)

    for req in requests:
        sched.add_request(
            req.request_id,
            req.prompt,
            SamplingParams(
                max_tokens=min(req.max_tokens, 100_000),
                seq_seed=req.seq_seed,
                stop_token_ids=(MOCK_EOS_TOKEN_ID,),
            ),
        )

    result = MockResult()
    steps = 0
    while sched.has_unfinished_requests:
        out = sched.schedule()
        assert not out.is_empty, (
            f"step {out.step_id}: scheduler made no progress but requests remain "
            f"(waiting={len(sched.waiting)}, running={len(sched.running)})"
        )
        sampled = mock.step(out)
        outputs = sched.update_from_output(sampled)
        for ro in outputs:
            result.outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                result.finish_reasons[ro.request_id] = ro.finish_reason
                result.cached_tokens[ro.request_id] = ro.num_cached_tokens
        _check_invariants(sched)
        result.trace.extend(sched.drain_trace())
        steps += 1
        assert steps < max_steps, "engine loop exceeded max_steps (livelock?)"
    result.num_steps = steps
    return result

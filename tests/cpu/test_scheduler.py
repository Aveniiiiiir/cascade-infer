"""Scheduler closed-loop tests (mock model): Phase 1 acceptance gate.

核心不变式（harness 每步校验）：块守恒、ref 计数 == block 表出现次数、
block 表恰好分配、RUNNING/WAITING 的 computed 语义、调度器永不空转。

输出不变式（批不变哲学的 Phase 1 版本）：mock 输出只依赖
(seed, 绝对位置) ⇒ 相同请求在任何池大小/抢占/chunk/批组合下输出相同。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cascade.sampling.rng import derive_uniform
from tests.cpu.harness import make_config, run_engine
from tests.cpu.mock_model import (
    MOCK_EOS_TOKEN_ID,
    MockModel,
    MockRequest,
    expected_tokens,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def make_requests(n: int, prompt_len: int, target: int, seed_base: int = 1000, salt: int = 0) -> list[MockRequest]:
    return [
        MockRequest(
            request_id=f"r{i}",
            prompt=[(i * 131 + j * 17 + salt) % 9000 + 10 for j in range(prompt_len)],
            seq_seed=seed_base + i,
            target_output_len=target,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------- basics


def test_basic_generation_matches_oracle() -> None:
    reqs = make_requests(4, prompt_len=100, target=50)
    result = run_engine(reqs, make_config(), num_blocks=64)
    for req in reqs:
        assert req.request_id in result.outputs, f"{req.request_id} 未完成"
        assert result.outputs[req.request_id] == expected_tokens(req, MockModel(reqs))
        assert result.finish_reasons[req.request_id] == "stop"  # EOS 收尾


def test_deterministic_across_runs() -> None:
    reqs = make_requests(6, prompt_len=80, target=40)
    a = run_engine(reqs, make_config(), num_blocks=64)
    b = run_engine(reqs, make_config(), num_blocks=64)
    assert a.outputs == b.outputs
    assert a.trace == b.trace


def test_chunked_prefill() -> None:
    # max_batched_tokens=96 < prompt 350 → 必然分块
    cfg = make_config(max_num_batched_tokens=96)
    reqs = make_requests(2, prompt_len=350, target=30)
    result = run_engine(reqs, cfg, num_blocks=64)
    for req in reqs:
        assert result.outputs[req.request_id] == expected_tokens(req, MockModel(reqs))
    # trace 里能看到多个 prefill 片段（P 前缀）
    prefill_chunks = [line for line in result.trace if ":P" in line]
    assert len(prefill_chunks) >= 4  # 350/96 → 每请求至少 4 块
    # 每步 token 预算不超
    for line in result.trace:
        tok = int(line.split("tok=")[1].split()[0])
        assert tok <= 96


def test_batch_budget_respected() -> None:
    cfg = make_config(max_num_batched_tokens=256, max_num_seqs=8)
    reqs = make_requests(12, prompt_len=70, target=20)
    result = run_engine(reqs, cfg, num_blocks=128)
    for line in result.trace:
        tok = int(line.split("tok=")[1].split()[0])
        assert tok <= 256
    assert all(r.request_id in result.outputs for r in reqs)


# --------------------------------------------------------------------- invariance


def test_output_invariant_to_pool_size_and_preemption() -> None:
    """小池触发抢占 + 重算，输出必须与大池运行完全一致。"""
    reqs = make_requests(5, prompt_len=120, target=200)
    big = run_engine(reqs, make_config(), num_blocks=256)
    small = run_engine(reqs, make_config(), num_blocks=14)  # 必然触发抢占
    for req in reqs:
        assert big.outputs[req.request_id] == small.outputs[req.request_id]
        assert big.outputs[req.request_id] == expected_tokens(req, MockModel(reqs))


def test_output_invariant_to_batching_order() -> None:
    """乱序入队 + 不同批组合 → 输出一致。"""
    reqs = make_requests(8, prompt_len=90, target=60)
    a = run_engine(reqs, make_config(max_num_batched_tokens=512), num_blocks=128)
    reversed_reqs = list(reversed(reqs))
    b = run_engine(reversed_reqs, make_config(max_num_batched_tokens=128), num_blocks=128)
    for req in reqs:
        assert a.outputs[req.request_id] == b.outputs[req.request_id]


# --------------------------------------------------------------------- prefix cache


def test_prefix_cache_hit_and_output_consistency() -> None:
    shared = [(7 * 131 + j * 17) % 9000 + 10 for j in range(256)]  # 4 个整块共享前缀
    req_a = MockRequest("a", shared + [11, 12, 13], seq_seed=1, target_output_len=5)
    req_b = MockRequest("b", shared + [21, 22, 23, 24], seq_seed=2, target_output_len=5)

    on = run_engine([req_a, req_b], make_config(), num_blocks=64)
    # b 在 a 之后运行：a 的满块留在缓存，b 命中 ≥ 3 块（第 4 块因尾部追加部分写共享）
    assert result_cached(on, "b") >= 3 * 64
    # 输出与无缓存运行一致（缓存正确性）
    off = run_engine([req_a, req_b], make_config(), num_blocks=64, enable_prefix_caching=False)
    assert on.outputs == off.outputs


def result_cached(result, request_id: str) -> int:
    return result.cached_tokens.get(request_id, 0)


def test_prefix_cache_saves_compute() -> None:
    shared = [j % 8000 + 10 for j in range(256)]
    req_a = MockRequest("a", shared + [1, 2, 3], seq_seed=1, target_output_len=3)
    req_b = MockRequest("b", shared + [4, 5, 6, 7], seq_seed=2, target_output_len=3)
    on = run_engine([req_a, req_b], make_config(), num_blocks=64)
    off = run_engine([req_a, req_b], make_config(), num_blocks=64, enable_prefix_caching=False)
    tok_on = total_prefill_tokens(on, "b")
    tok_off = total_prefill_tokens(off, "b")
    assert tok_on < tok_off  # 命中缓存省 prefill 计算
    assert tok_off == len(req_b.prompt)


def total_prefill_tokens(result, request_id: str) -> int:
    total = 0
    for line in result.trace:
        for part in line.split():
            if part.startswith(f"{request_id}:P"):
                total += int(part.split(":P")[1].rstrip("*"))
    return total


# --------------------------------------------------------------------- stop / limits


def test_max_tokens_length_cap() -> None:
    req = MockRequest("r", list(range(100)), seq_seed=5, target_output_len=500, max_tokens=7)
    result = run_engine([req], make_config(), num_blocks=64)
    assert result.finish_reasons["r"] == "length"
    assert len(result.outputs["r"]) == 7


def test_model_len_cap() -> None:
    cfg = make_config(max_model_len=110)  # runtime = 111
    req = MockRequest("r", list(range(100)), seq_seed=5, target_output_len=500)
    result = run_engine([req], cfg, num_blocks=64)
    assert result.finish_reasons["r"] == "length"
    # prompt 100 + 输出直到 num_tokens+1 > 111 → 输出 10 个
    assert len(result.outputs["r"]) == 10


def test_abort_mid_run() -> None:
    reqs = make_requests(3, prompt_len=100, target=300)
    from cascade.config import EngineConfig
    from cascade.core.block_manager import BlockManager
    from cascade.core.prefix_cache import PrefixCache
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config()
    bm = BlockManager(64)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(reqs)
    for r in reqs:
        sched.add_request(r.request_id, r.prompt, SamplingParams(seq_seed=r.seq_seed, max_tokens=1000, stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    # 跑几步后 abort r1
    for _ in range(10):
        out = sched.schedule()
        sched.update_from_output(mock.step(out))
    assert sched.abort("r1") is True
    steps = 0
    while sched.has_unfinished_requests:
        out = sched.schedule()
        outputs = sched.update_from_output(mock.step(out))
        for ro in outputs:
            if ro.finish_reason:
                assert ro.request_id != "r1" or ro.finish_reason == "abort"
        steps += 1
        assert steps < 5000
    assert not sched.abort("r1")  # 已结束


def test_prompt_too_large_rejected() -> None:
    from cascade.config import EngineConfig
    from cascade.core.block_manager import BlockManager
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config()
    bm = BlockManager(2)  # 2 * 64 = 128 token 容量
    sched = Scheduler(cfg, bm, None)
    with pytest.raises(ValueError, match="needs .* blocks"):
        sched.add_request("big", list(range(200)), SamplingParams())


# --------------------------------------------------------------------- preemption semantics


def test_so_request_preempted_is_killed() -> None:
    """SO 请求被抢占必判死（DeepInfer 语义）→ FINISHED_PREEMPTED。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.prefix_cache import PrefixCache
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config()
    bm = BlockManager(12)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    so_req = MockRequest("so", list(range(100)), seq_seed=1, target_output_len=400)
    lates = [MockRequest(f"late{i}", list(range(1000 + 50 * i, 1100 + 50 * i)), seq_seed=2 + i, target_output_len=80) for i in range(4)]
    mock = MockModel([so_req, *lates])
    sched.add_request(
        "so", so_req.prompt,
        SamplingParams(seq_seed=1, max_tokens=1000, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        has_so=True,
    )
    # 先让 so 跑起来（开始 decode）
    for _ in range(5):
        out = sched.schedule()
        sched.update_from_output(mock.step(out))
    assert "so" in [s.request_id for s in sched.running]
    # 迟到请求涌入 → 池紧张 → 抢占 output 最长的 so（SO 必判死）
    for r in lates:
        sched.add_request(r.request_id, r.prompt, SamplingParams(seq_seed=r.seq_seed, max_tokens=1000, stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    finished: dict[str, str] = {}
    steps = 0
    while sched.has_unfinished_requests:
        out = sched.schedule()
        for ro in sched.update_from_output(mock.step(out)):
            if ro.finish_reason:
                finished[ro.request_id] = ro.finish_reason
        steps += 1
        assert steps < 5000
    assert finished.get("so") == "preempt"
    for r in lates:
        assert finished.get(r.request_id) == "stop"


# --------------------------------------------------------------------- golden trace


def test_scheduler_trace_golden() -> None:
    reqs = [
        MockRequest("g0", [(i * 17) % 9000 + 10 for i in range(300)], 11, 120),
        MockRequest("g1", [(i * 17) % 9000 + 10 for i in range(300)] + [777, 888], 12, 90),
        MockRequest("g2", [(i * 29) % 9000 + 10 for i in range(90)], 13, 200),
        MockRequest("g3", [(i * 41) % 9000 + 10 for i in range(150)], 14, 160),
        MockRequest("g4", [(i * 53) % 9000 + 10 for i in range(500)], 15, 40),
    ]
    cfg = make_config(max_num_batched_tokens=512, max_num_seqs=8)
    result = run_engine(reqs, cfg, num_blocks=30)  # 小池 → 触发抢占
    trace_text = "\n".join(result.trace) + "\n"
    golden = GOLDEN_DIR / "scheduler_trace.txt"
    if os.environ.get("CASCADE_REGEN_GOLDEN", "0") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(trace_text, encoding="utf-8")
    assert golden.exists(), "golden 文件缺失：先 CASCADE_REGEN_GOLDEN=1 pytest 生成"
    expected = golden.read_text(encoding="utf-8")
    assert trace_text == expected, "调度 trace 与 golden 不一致（调度行为回归）"
    # 输出仍然正确（trace 变化时最关键的底线）
    mock = MockModel(reqs)
    for req in reqs:
        assert result.outputs[req.request_id] == expected_tokens(req, mock)

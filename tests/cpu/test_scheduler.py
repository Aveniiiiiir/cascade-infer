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

from cascade.core.sequence import SequenceStatus
from cascade.sampling.rng import derive_uniform
from tests.cpu.harness import make_config, run_engine
from tests.cpu.mock_model import (
    MOCK_EOS_TOKEN_ID,
    MockModel,
    MockRequest,
    MockResult,
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


def _run_sequential(reqs: list, cfg, num_blocks: int, enable_prefix: bool = True):
    """串行运行（a 完成后才入队 b）——前缀命中测试的前提：多 prompt 打包
    （一步多 prefill）后，同时入队的相同前缀请求会在空缓存上并发 prefill。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.prefix_cache import PrefixCache
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg.scheduler.enable_prefix_caching = enable_prefix
    bm = BlockManager(num_blocks)
    pc = PrefixCache(bm, cfg.cache.block_size) if enable_prefix else None
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(reqs)
    result = MockResult()
    for req in reqs:
        sched.add_request(
            req.request_id, req.prompt,
            SamplingParams(max_tokens=10_000, seq_seed=req.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        )
        while sched.has_unfinished_requests:
            out = sched.schedule()
            outputs = sched.update_from_output(mock.step(out))
            for ro in outputs:
                result.outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
                if ro.finish_reason is not None:
                    result.finish_reasons[ro.request_id] = ro.finish_reason
                    result.cached_tokens[ro.request_id] = ro.num_cached_tokens
            result.trace.extend(sched.drain_trace())
    return result


def test_prefix_cache_hit_and_output_consistency() -> None:
    shared = [(7 * 131 + j * 17) % 9000 + 10 for j in range(256)]  # 4 个整块共享前缀
    req_a = MockRequest("a", shared + [11, 12, 13], seq_seed=1, target_output_len=5)
    req_b = MockRequest("b", shared + [21, 22, 23, 24], seq_seed=2, target_output_len=5)

    on = _run_sequential([req_a, req_b], make_config(), 64)
    # b 在 a 之后运行：a 的满块留在缓存，b 命中 4 块（259/260 token 的前 256）
    assert result_cached(on, "b") >= 3 * 64
    # 输出与无缓存运行一致（缓存正确性）
    off = _run_sequential([req_a, req_b], make_config(), 64, enable_prefix=False)
    assert on.outputs == off.outputs


def result_cached(result, request_id: str) -> int:
    return result.cached_tokens.get(request_id, 0)


def test_prefix_cache_saves_compute() -> None:
    shared = [j % 8000 + 10 for j in range(256)]
    req_a = MockRequest("a", shared + [1, 2, 3], seq_seed=1, target_output_len=3)
    req_b = MockRequest("b", shared + [4, 5, 6, 7], seq_seed=2, target_output_len=3)
    on = _run_sequential([req_a, req_b], make_config(), 64)
    off = _run_sequential([req_a, req_b], make_config(), 64, enable_prefix=False)
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


# --------------------------------------------------------------------- review findings 回归测试


def test_duplicate_prompts_full_prefix_hit_clamped() -> None:
    """审查 #3：两个完全相同的 prompt（串行入队）——第二个命中全量前缀，
    必须 clamp 到至少留 1 个新 token（禁止 0-token prefill 行）。"""
    prompt = [(i * 19) % 9000 + 10 for i in range(128)]  # 恰好 2 个整块
    a = MockRequest("a", prompt, 1, 5)
    b = MockRequest("b", list(prompt), 2, 5)
    result = _run_sequential([a, b], make_config(), 64)
    # b 的 prefill 行全部 ≥ 1 token，且被 clamp 后只算 64 个新 token
    b_chunks = [
        int(part.split(":P")[1].rstrip("*"))
        for line in result.trace
        for part in line.split()
        if part.startswith("b:P")
    ]
    assert b_chunks and all(c >= 1 for c in b_chunks)
    assert sum(b_chunks) == 128 - 64  # clamp 到 1 个整块
    # 输出与 oracle 一致（缓存 clamp 不影响正确性）
    mock = MockModel([a, b])
    assert result.outputs["b"] == expected_tokens(b, mock)


def test_pool_exhaustion_decode_finished_ignored() -> None:
    """审查 #2：decode 增长超过池容量 → FINISHED_IGNORED（绝不活锁）。"""
    req = MockRequest("r", list(range(100)), seq_seed=9, target_output_len=500, max_tokens=100_000)
    result = run_engine([req], make_config(), num_blocks=3)  # 3 块 = 192 token 容量
    assert result.finish_reasons.get("r") == "ignored"
    assert result.num_steps < 200  # 引擎正常收尾，没有无限空转


def test_recompute_footprint_exceeds_pool_finished_ignored() -> None:
    """审查 #2（prefill 侧）：重算足迹（prompt+已生成）超池 → FINISHED_IGNORED。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config()
    bm = BlockManager(3)
    sched = Scheduler(cfg, bm, None)
    sp = SamplingParams(seq_seed=1, max_tokens=1000, stop_token_ids=(MOCK_EOS_TOKEN_ID,))
    seq = sched.add_request("r", list(range(100)), sp)  # prompt 2 块，放得下
    seq.append_output_tokens(list(range(100)))  # 重算场景：已生成 100 → 足迹 200 > 3 块
    out = sched.schedule()
    outputs = sched.update_from_output({})
    assert all(s.num_new_tokens >= 1 for s in out.scheduled) or not out.scheduled
    assert outputs and outputs[0].finish_reason == "ignored"
    assert not sched.has_unfinished_requests
    bm.debug_check_consistency()


def test_multiple_whole_prompts_packed_per_step() -> None:
    """审查 #5：预算充足时多个完整 prompt 打包进同一步。"""
    reqs = make_requests(3, prompt_len=100, target=20)
    result = run_engine(reqs, make_config(), num_blocks=64)
    first = result.trace[0]
    n_prefill_rows = sum(1 for p in first.split() if ":P" in p)
    assert n_prefill_rows == 3  # 3 个完整 prompt 一步打包
    for req in reqs:
        assert result.outputs[req.request_id] == expected_tokens(req, MockModel(reqs))


def test_schedule_reentry_guard() -> None:
    """审查 #7：连续 schedule 不夹 update → 显式 AssertionError。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    sched = Scheduler(make_config(), BlockManager(16), None)
    sched.add_request("r", list(range(10)), SamplingParams())
    sched.schedule()
    with pytest.raises(AssertionError, match="重入"):
        sched.schedule()


def test_abort_in_limbo_window() -> None:
    """审查 #4：schedule→update 窗口内（队列外悬空态）abort 仍生效，
    update 不崩溃，块守恒。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.prefix_cache import PrefixCache
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config()
    bm = BlockManager(16)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel([MockRequest("x", list(range(10)), 1, 5)])
    sched.add_request("x", list(range(10)), SamplingParams(seq_seed=1, max_tokens=100, stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    out = sched.schedule()
    assert any(s.completes_prefill for s in out.scheduled)
    # 悬空窗口：x 已 pop 出 waiting、尚未进 running
    assert sched.abort("x") is True  # _find 扫到 _last_output 里的悬空 seq
    assert not sched.has_unfinished_requests
    outputs = sched.update_from_output({"x": [42]})  # 窗口内采样结果被丢弃
    assert any(ro.finish_reason == "abort" for ro in outputs)
    assert all(ro.request_id != "x" or ro.finish_reason == "abort" for ro in outputs)
    bm.debug_check_consistency()
    pc.debug_check_consistency()


def test_cow_clone_shared_partial_block() -> None:
    """审查 #6：fork 共享半写尾块 → 写入者 COW 克隆；容量不足 → 状态不变。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config(block_size=8)
    bm = BlockManager(4)
    sched = Scheduler(cfg, bm, None)
    sp = SamplingParams(seq_seed=1, max_tokens=100, stop_token_ids=(MOCK_EOS_TOKEN_ID,))
    parent = sched.add_request("p", list(range(10)), sp)  # 2 块
    child = sched.add_request("c", list(range(10)), sp)
    # 手工构造 fork 态：parent RUNNING（computed=9 = 半写 b2），child 共享其块
    b1, b2 = bm.allocate(), bm.allocate()
    bm.retain(b1)
    bm.retain(b2)  # child 的引用
    parent.block_table = [b1, b2]
    parent.num_tokens_computed = 9
    parent.status = SequenceStatus.RUNNING
    sched.waiting.remove(parent)  # 手工晋升 running：从 waiting 移除
    sched.running.append(parent)
    child.block_table = [b1, b2]
    child.num_tokens_computed = 9  # WAITING，重算中（共享半写尾块）
    # 池 4 块：b1/b2 已用，free=2 → parent decode 先调度（COW b2→b3）
    out = sched.schedule()
    assert parent.block_table[-1] != b2 and child.block_table[-1] == b2
    assert out.cow_copies and out.cow_copies[0][1] == b2
    assert bm.ref_count(b2) == 1  # 只剩 child 持有
    assert bm.ref_count(parent.block_table[-1]) == 1
    bm.debug_check_consistency()

    # 容量不足变体：池恰好耗尽 → COW 失败前置，状态不变、不崩溃
    bm2 = BlockManager(3)
    sched2 = Scheduler(cfg, bm2, None)
    p2 = sched2.add_request("p", list(range(10)), sp)
    c2 = sched2.add_request("c", list(range(10)), sp)
    b1, b2 = bm2.allocate(), bm2.allocate()
    bm2.retain(b1)
    bm2.retain(b2)
    p2.block_table = [b1, b2]
    p2.num_tokens_computed = 9
    p2.status = SequenceStatus.RUNNING
    sched2.waiting.remove(p2)  # 手工晋升 running
    sched2.running.append(p2)
    c2.block_table = [b1, b2]
    c2.num_tokens_computed = 9
    out2 = sched2.schedule()  # free=1 → decode(p2) COW 需要 1 块：成功；随后 c2 chunk 无需新块
    # p2 COW 后 free=0；c2 的尾块 b2 现在 ref==1（p2 已克隆）→ 无需 COW，need=0 → 调度成功
    assert any(s.seq.request_id == "c" or s.seq.request_id == "p" for s in out2.scheduled)
    bm2.debug_check_consistency()


def test_cow_exhaustion_preempts_and_recovers() -> None:
    """审查 #6(a)：池耗尽时 COW 不裸抛 BlockAllocationError。

    语义推演（pool=2，parent/child 共享半写 b2，free=0）：
    - step1: p 的 decode COW 失败（抢占排除自身 → 跳过，不崩溃）；
      c 的 prefill 触发对 p 的 RECOMPUTE 抢占（b2 ref 2→1）后自身容量仍不足 →
      空步（合法：下一步 c 的尾块已私有化）。
    - step2: c 无需 COW（b2 ref==1），P1 调度成功。
    """
    from cascade.core.block_manager import BlockManager
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams

    cfg = make_config(block_size=8)
    bm = BlockManager(2)  # b1/b2 即全部
    sched = Scheduler(cfg, bm, None)
    sp = SamplingParams(seq_seed=1, max_tokens=100, stop_token_ids=(MOCK_EOS_TOKEN_ID,))
    parent = sched.add_request("p", list(range(10)), sp)
    child = sched.add_request("c", list(range(10)), sp)
    b1, b2 = bm.allocate(), bm.allocate()
    bm.retain(b1)
    bm.retain(b2)
    parent.block_table = [b1, b2]
    parent.num_tokens_computed = 9
    parent.status = SequenceStatus.RUNNING
    sched.waiting.remove(parent)  # 手工晋升 running：从 waiting 移除
    sched.running.append(parent)
    child.block_table = [b1, b2]
    child.num_tokens_computed = 9
    # step1：无崩溃；p 被重算抢占（其 decode COW 失败 + c 的容量需求）
    out1 = sched.schedule()
    assert all(s.seq.request_id != "p" for s in out1.scheduled)
    assert not out1.cow_copies  # COW 前置容量检查：失败不留半途状态
    sched.update_from_output({})
    assert parent.status is SequenceStatus.WAITING  # RECOMPUTE 抢占
    # step2：b2 已私有化（ref==1）→ c 正常调度
    out2 = sched.schedule()
    assert any(s.seq.request_id == "c" for s in out2.scheduled)
    assert child.block_table[-1] == b2  # 无需克隆：唯一持有者
    bm.debug_check_consistency()


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

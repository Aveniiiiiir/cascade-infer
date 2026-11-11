"""ZOS 零开销调度协议的调度器级测试（CPU mock 驱动，本机可跑）。

三段式顺序：schedule(N) → update(N−1)（1 步 lag，pending 深度 ≤ 2）。
核心不变式：**ZOS 顺序与同步顺序产出完全相同的输出**（协议正确性）。
"""

from __future__ import annotations

from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.sequence import RequestOutput
from tests.cpu.harness import make_config, run_engine
from tests.cpu.mock_model import MOCK_EOS_TOKEN_ID, MockModel, MockRequest, expected_tokens


def make_requests(n: int, prompt_len: int, target: int, seed_base: int = 500) -> list[MockRequest]:
    return [
        MockRequest(
            request_id=f"z{i}",
            prompt=[(i * 97 + j * 13 + 7) % 9000 + 10 for j in range(prompt_len)],
            seq_seed=seed_base + i,
            target_output_len=target,
        )
        for i in range(n)
    ]


def _run_zos(requests: list[MockRequest], num_blocks: int, block_size: int = 64):
    """ZOS 顺序闭环：schedule(N) → update(N−1)；返回 (outputs, finish, trace)。"""
    from cascade.sampling_params import SamplingParams
    from cascade.core.scheduler import Scheduler

    cfg = make_config(block_size=block_size)
    bm = BlockManager(num_blocks)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(requests)
    for r in requests:
        sched.add_request(
            r.request_id, r.prompt,
            SamplingParams(max_tokens=10_000, seq_seed=r.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        )
    outputs: dict[str, list[int]] = {}
    finish: dict[str, str] = {}
    trace: list[str] = []
    prev_sampled: dict[str, list[int]] | None = None
    steps = 0
    while sched.has_unfinished_requests:
        out = sched.schedule()  # schedule(N)（乐观推进）
        if prev_sampled is not None:
            for ro in sched.update_from_output(prev_sampled):  # 提交 N−1
                outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
                if ro.finish_reason is not None:
                    finish[ro.request_id] = ro.finish_reason
        prev_sampled = mock.step(out)
        trace.extend(sched.drain_trace())
        steps += 1
        assert steps < 20_000
    # 尾部冲刷
    if prev_sampled is not None:
        for ro in sched.update_from_output(prev_sampled):
            outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish[ro.request_id] = ro.finish_reason
    assert not sched.has_unfinished_requests
    bm.debug_check_consistency()
    pc.debug_check_consistency()
    return outputs, finish, trace


def test_zos_order_matches_sync_outputs() -> None:
    """协议正确性：ZOS 顺序（1 步 lag）与同步顺序输出逐 token 一致。"""
    reqs = make_requests(6, prompt_len=130, target=60)
    sync = run_engine(reqs, make_config(), num_blocks=64)
    zos_out, zos_fin, _trace = _run_zos(reqs, 64)
    assert zos_out == sync.outputs
    assert zos_fin == sync.finish_reasons
    # 且与 oracle 一致
    mock = MockModel(reqs)
    for r in reqs:
        assert zos_out[r.request_id] == expected_tokens(r, mock)


def test_zos_order_with_preemption_and_small_pool() -> None:
    """小池触发抢占 + 重算下，ZOS 顺序仍产出正确输出。"""
    reqs = make_requests(5, prompt_len=120, target=200)
    big = run_engine(reqs, make_config(), num_blocks=256)
    zos_out, zos_fin, _trace = _run_zos(reqs, 14)  # 必然触发抢占
    for r in reqs:
        assert zos_out[r.request_id] == big.outputs[r.request_id]
        assert zos_out[r.request_id] == expected_tokens(r, MockModel(reqs))


def test_zos_pending_depth_invariant() -> None:
    """pending 深度 ≤ 2：合法 lag 两次；第三次 schedule（入口深度 2）拒绝。"""
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams
    import pytest

    cfg = make_config()
    sched = Scheduler(cfg, BlockManager(32), None)
    sched.add_request("r", list(range(10)), SamplingParams())
    sched.schedule()  # 深度 1
    sched.schedule()  # 深度 2（ZOS 稳态）
    with pytest.raises(AssertionError, match="重入"):
        sched.schedule()

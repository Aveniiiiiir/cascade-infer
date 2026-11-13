"""调度器新增功能测试（审查修复项的 CPU 门禁）：
- 多 token stop 序列（#14）：单元（步内截断）+ 引擎级（完成点检测）；
- flush_final_outputs / fail_all_inflight（#2/#4 配套）。
"""

from __future__ import annotations

from cascade.core.block_manager import BlockManager
from cascade.core.scheduler import Scheduler
from cascade.core.sequence import Sequence, SequenceStatus
from cascade.sampling_params import SamplingParams
from tests.cpu.harness import _check_invariants, make_config
from tests.cpu.mock_model import MOCK_EOS_TOKEN_ID, MockModel, MockRequest, MockResult, expected_tokens


def _make_sched(num_blocks: int = 128, **cfg_kw) -> Scheduler:
    cfg = make_config(**cfg_kw)
    return Scheduler(cfg, BlockManager(num_blocks), None)


def _single_req_engine(sched: Scheduler, stop_seqs, mock_out_len: int = 50):
    req = MockRequest("r1", prompt=[1, 2, 3, 4, 5], seq_seed=42, target_output_len=mock_out_len)
    sp = SamplingParams(
        max_tokens=10_000, seq_seed=req.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,),
        stop_token_seqs=stop_seqs,
    )
    sched.add_request("r1", req.prompt, sp)
    return MockModel([req]), req


def _drain(sched: Scheduler, mock: MockModel) -> MockResult:
    """同步序闭环（引擎 step 的 CPU 等价）。"""
    result = MockResult()
    while sched.has_unfinished_requests:
        out = sched.schedule()
        sampled = mock.step(out)
        outputs = sched.update_from_output(sampled)
        for ro in outputs:
            result.outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                result.finish_reasons[ro.request_id] = ro.finish_reason
        _check_invariants(sched)
    return result


# ------------------------------------------------------------------ _truncate 单元


def test_truncate_stop_seq_within_step() -> None:
    """多 token 提交（MTP 形态）的步内截断：stop 序列整体落在 new tokens 内。"""
    sched = _make_sched()
    seq = Sequence("u", [1, 2, 3], SamplingParams(max_tokens=100, stop_token_seqs=((60, 61),)).snapshot())
    # new = [55, 60, 61, 62]：序列 (60,61) 在其中 → 截到 55
    final, finish = sched._truncate_new_tokens(seq, [55, 60, 61, 62])
    assert final == [55]
    assert finish is SequenceStatus.FINISHED_STOPPED
    # 序列起于本步首 token → 空提交 + 判停（合法：前缀在先前步已发）
    final, finish = sched._truncate_new_tokens(Sequence("u", [1, 2, 3], SamplingParams(max_tokens=100, stop_token_seqs=((55, 60),)).snapshot()), [55, 60, 61])
    assert final == []
    assert finish is SequenceStatus.FINISHED_STOPPED
    # 不命中 → 原样
    seq3 = Sequence("u", [1, 2, 3], SamplingParams(max_tokens=100, stop_token_seqs=((9, 9),)).snapshot())
    final, finish = sched._truncate_new_tokens(seq3, [55, 60, 61])
    assert final == [55, 60, 61]
    assert finish is None


def test_truncate_stop_seq_uses_committed_output_tail() -> None:
    """后缀匹配覆盖「已提交输出 ‖ 本步新 token」整体（跨步序列）。"""
    sched = _make_sched()
    seq = Sequence("u", [1, 2, 3], SamplingParams(max_tokens=100, stop_token_seqs=((7, 8),)).snapshot())
    seq.append_output_tokens([6, 7])  # 已提交的先前步输出
    final, finish = sched._truncate_new_tokens(seq, [8])
    # full = [6,7,8]，序列 (7,8) 完成 → cut=1 < 已提交长度 2 → 空提交
    assert final == []
    assert finish is SequenceStatus.FINISHED_STOPPED


# ------------------------------------------------------------------ 引擎级（单 token/步）


def test_stop_seq_detected_at_completion() -> None:
    """单 token/步提交：序列 (t1, t2) 在 t2 到达的步命中 → 判停；
    已发出的 t0/t1 保留（文档化限制：不回收已流式发出的 stop 文本前缀）。"""
    sched = _make_sched()
    mock, req = _single_req_engine(sched, stop_seqs=None)
    expected = expected_tokens(req, mock)
    sched = _make_sched()
    mock, req = _single_req_engine(sched, stop_seqs=((expected[1], expected[2]),))
    result = _drain(sched, mock)
    assert result.outputs["r1"] == expected[:2]
    assert result.finish_reasons["r1"] == "stop"


def test_stop_seq_prefix_emitted_before_completion() -> None:
    """序列 (t2, t3)：完成前 t0..t2 已流式发出 → 输出 = expected[:3]。"""
    sched = _make_sched()
    mock, req = _single_req_engine(sched, stop_seqs=None)
    expected = expected_tokens(req, mock)
    sched = _make_sched()
    mock, req = _single_req_engine(sched, stop_seqs=((expected[2], expected[3]),))
    result = _drain(sched, mock)
    assert result.outputs["r1"] == expected[:3]
    assert result.finish_reasons["r1"] == "stop"


def test_stop_seq_never_matched_runs_to_mock_eos() -> None:
    sched = _make_sched()
    mock, req = _single_req_engine(sched, stop_seqs=((999999, 888888),))
    result = _drain(sched, mock)
    assert result.outputs["r1"] == expected_tokens(req, mock)
    assert result.finish_reasons["r1"] == "stop"  # mock EOS（target_output_len）


# ------------------------------------------------------------------ flush / fail_all


def test_flush_final_outputs_delivers_abort_receipt() -> None:
    """abort 的最终回执不再依赖下一次 update（空调度步也能冲刷）。"""
    sched = _make_sched()
    sched.add_request("r1", [1, 2, 3], SamplingParams(seq_seed=1, max_tokens=10,
                                                      stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    sched.add_request("r2", [4, 5], SamplingParams(seq_seed=2, max_tokens=10,
                                                   stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    sched.schedule()
    assert sched.abort("r1")
    flushed = sched.flush_final_outputs()
    assert [ro.request_id for ro in flushed] == ["r1"]
    assert flushed[0].finish_reason == "abort"
    assert sched.flush_final_outputs() == []  # 幂等


def test_fail_all_inflight_resets_engine_state() -> None:
    """引擎步异常的兜底：判死全部在途（含已调度未提交）、清空 pending、
    块全部释放 —— 之后 has_unfinished 为 False，可重新 add_request。"""
    cfg = make_config()
    block_mgr = BlockManager(128)
    sched = Scheduler(cfg, block_mgr, None)
    sched.add_request("r1", [1, 2, 3], SamplingParams(seq_seed=1, max_tokens=10,
                                                      stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    sched.add_request("r2", [4, 5, 6, 7], SamplingParams(seq_seed=2, max_tokens=10,
                                                         stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    sched.schedule()  # pending 深度 1（未提交）
    outputs = sched.fail_all_inflight()
    assert {ro.request_id for ro in outputs} == {"r1", "r2"}
    assert all(ro.finish_reason == "abort" for ro in outputs)
    assert not sched.has_unfinished_requests
    assert not sched._pending_outputs
    block_mgr.debug_check_consistency()
    assert block_mgr.num_allocated_blocks == 0
    # 引擎可继续服务新请求
    sched.add_request("r3", [8, 9], SamplingParams(seq_seed=3, max_tokens=5,
                                                   stop_token_ids=(MOCK_EOS_TOKEN_ID,)))
    assert sched.has_unfinished_requests
    assert not sched.schedule().is_empty


# ------------------------------------------------------------------ chunked 闭环


def test_stopseq_chunked_multi_request() -> None:
    """chunked prefill + 双请求 + stop 序列：闭环不变量保持，截断语义正确。"""
    sched = _make_sched(num_blocks=256, max_num_batched_tokens=16)  # 强制 chunked
    reqs = [
        MockRequest("a", prompt=list(range(10)), seq_seed=7, target_output_len=20),
        MockRequest("b", prompt=list(range(10, 25)), seq_seed=8, target_output_len=20),
    ]
    mock = MockModel(reqs)
    exp_a = expected_tokens(reqs[0], mock)
    exp_b = expected_tokens(reqs[1], mock)
    stop_a = (exp_a[2], exp_a[3])
    for r in reqs:
        sched.add_request(
            r.request_id, r.prompt,
            SamplingParams(max_tokens=10_000, seq_seed=r.seq_seed,
                           stop_token_ids=(MOCK_EOS_TOKEN_ID,),
                           stop_token_seqs=(stop_a,)),
        )
    result = _drain(sched, mock)
    assert result.outputs["a"] == exp_a[:3]  # 完成点判停（见上方语义注记）
    assert result.finish_reasons["a"] == "stop"
    assert result.outputs["b"] == exp_b  # b 不含 stop_a → 跑到 mock EOS
    assert result.finish_reasons["b"] == "stop"

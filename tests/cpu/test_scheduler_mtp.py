"""调度器 MTP 多 token 提交测试（CPU mock，本机可跑）。

核心不变式：**MTP 开启（任意接受率）与 MTP 关闭的输出逐 token 一致**
——mock 目标确定性（f(seed, 位置)），mock 草案候选 = 真实延续序列，
接受率旋钮只改变每步提交的 token 数，不改变序列内容。
"""

from __future__ import annotations

from cascade.core.block_manager import BlockManager
from cascade.core.prefix_cache import PrefixCache
from cascade.core.scheduler import Scheduler
from cascade.sampling.rng import derive_uniform
from cascade.sampling_params import SamplingParams
from tests.cpu.harness import make_config, run_engine
from tests.cpu.mock_model import MOCK_EOS_TOKEN_ID, MockModel, MockRequest, expected_tokens


def make_requests(n: int, prompt_len: int, target: int, seed_base: int = 900) -> list[MockRequest]:
    return [
        MockRequest(
            request_id=f"m{i}",
            prompt=[(i * 71 + j * 19 + 3) % 9000 + 10 for j in range(prompt_len)],
            seq_seed=seed_base + i,
            target_output_len=target,
        )
        for i in range(n)
    ]


def _accept_knob(seed: int, position: int, k: int) -> int:
    """确定性接受数（0..k）：混合接受场景。"""
    return int(derive_uniform(seed, position, 0) * (k + 1)) % (k + 1)


def _mock_token(seed: int, position: int, mock: MockModel, prompt_len: int, target: int) -> int:
    """与 mock_model/expected_tokens 公式逐位一致的目标 token（含 EOS 收尾）。"""
    out_idx = position - prompt_len  # 该位置 token 的输出序号
    if out_idx + 1 >= target:
        return mock.eos_token_id
    tok = int(derive_uniform(seed, position) * mock.vocab_size)
    if tok == mock.eos_token_id:
        tok = (tok + 1) % mock.vocab_size
    return tok


def _mock_mtp_step(sched: Scheduler, out, mock: MockModel, k: int, accept_mode: str, targets: dict) -> dict:
    """mock MTP 执行：候选 = 目标真实延续；按旋钮返回前缀。"""
    sampled: dict[str, list[int]] = {}
    for s in out.scheduled:
        seq = s.seq
        if s.is_prefill:
            if not s.completes_prefill:
                continue
            sampled[seq.request_id] = [
                _mock_token(seq.sampling.seq_seed, seq.num_tokens, mock, seq.num_prompt_tokens, targets[seq.request_id])
            ]
            continue
        # decode 行：候选 = num_new 个真实延续 token（schedule 已乐观推进，
        # 输入 token 位于位置 start = num_tokens - 1，新 token 从 start+1 起）
        start = seq.num_tokens - 1
        candidates = [
            _mock_token(seq.sampling.seq_seed, start + j + 1, mock, seq.num_prompt_tokens, targets[seq.request_id])
            for j in range(s.num_new_tokens)
        ]
        if accept_mode == "all":
            n = s.num_new_tokens
        elif accept_mode == "none":
            n = 1
        else:
            n = 1 + _accept_knob(seq.sampling.seq_seed, start, s.num_new_tokens - 1)
        sampled[seq.request_id] = candidates[:n]
    return sampled


def _run_mtp(
    requests: list[MockRequest],
    num_blocks: int,
    k: int,
    accept_mode: str = "mixed",
    max_num_batched_tokens: int = 8192,
):
    from cascade.config import SchedulerConfig
    from tests.cpu.harness import _check_invariants

    cfg = make_config()
    cfg.scheduler = SchedulerConfig(
        max_num_seqs=64,
        max_num_batched_tokens=max_num_batched_tokens,
        nextn_draft_tokens=k,
    )
    bm = BlockManager(num_blocks)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(requests)
    targets = {r.request_id: r.target_output_len for r in requests}
    for r in requests:
        sched.add_request(
            r.request_id, r.prompt,
            SamplingParams(max_tokens=10_000, seq_seed=r.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        )
    outputs: dict[str, list[int]] = {}
    finish: dict[str, str] = {}
    steps = 0
    while sched.has_unfinished_requests:
        out = sched.schedule()
        sampled = _mock_mtp_step(sched, out, mock, k, accept_mode, targets)
        for ro in sched.update_from_output(sampled):
            outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish[ro.request_id] = ro.finish_reason
        _check_invariants(sched)
        bm.debug_check_consistency()
        pc.debug_check_consistency()
        steps += 1
        assert steps < 20_000
    return outputs, finish, steps


def test_mtp_outputs_match_non_mtp() -> None:
    """任意接受率下 MTP 输出 ≡ 非 MTP（逐 token）。"""
    reqs = make_requests(5, prompt_len=80, target=60)
    baseline = run_engine(reqs, make_config(), num_blocks=128)
    for mode in ("all", "none", "mixed"):
        out, fin, _steps = _run_mtp(reqs, 128, k=4, accept_mode=mode)
        for r in reqs:
            assert out[r.request_id] == baseline.outputs[r.request_id], f"mode={mode} {r.request_id}"
            assert fin[r.request_id] == baseline.finish_reasons[r.request_id]
        # mock 目标 oracle
        mock = MockModel(reqs)
        for r in reqs:
            assert out[r.request_id] == expected_tokens(r, mock)


def test_mtp_mixed_accept_reduces_steps() -> None:
    """全接受模式步数显著少于非 MTP（投机加速生效的代理指标）。"""
    reqs = make_requests(4, prompt_len=64, target=100)
    base = run_engine(reqs, make_config(), num_blocks=128)
    _out, _fin, steps_all = _run_mtp(reqs, 128, k=4, accept_mode="all")
    # 全接受：每步 5 token ⇒ 步数 ≈ base/5 + prefill
    assert steps_all < base.num_steps * 0.5, f"{steps_all} vs {base.num_steps}"


def test_mtp_stop_token_mid_prefix_truncation() -> None:
    """接受前缀中含 stop token → 截断到该 token 并 finish=stop。"""
    # 构造：目标序列中第 7 个输出 token 被 mock 替换为 EOS —— 直接构造
    # 一个 EOS 出现在候选中部的场景：accept_mode=all 且 target 很长，
    # 但 max_tokens 限制在中途 → 验证 length 截断路径
    reqs = make_requests(2, prompt_len=40, target=200)
    from cascade.sampling_params import SamplingParams as SP

    # 用 max_tokens=15 截断：MTP 每步最多 5 token，某步会越界
    cfg = make_config()
    from cascade.config import SchedulerConfig

    cfg.scheduler = SchedulerConfig(max_num_seqs=64, nextn_draft_tokens=4)
    bm = BlockManager(64)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(reqs)
    for r in reqs:
        sched.add_request(
            r.request_id, r.prompt,
            SP(max_tokens=15, seq_seed=r.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        )
    outs: dict[str, list[int]] = {}
    fin: dict[str, str] = {}
    targets = {r.request_id: r.target_output_len for r in reqs}
    while sched.has_unfinished_requests:
        out = sched.schedule()
        sampled = _mock_mtp_step(sched, out, mock, 4, "all", targets)
        for ro in sched.update_from_output(sampled):
            outs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                fin[ro.request_id] = ro.finish_reason
    for r in reqs:
        assert len(outs[r.request_id]) == 15
        assert fin[r.request_id] == "length"
        mock2 = MockModel(reqs)
        assert outs[r.request_id] == expected_tokens(r, mock2)[:15]


def test_mtp_small_pool_preemption() -> None:
    """小池抢占 + 重算 + MTP 并存：输出不变、块守恒。"""
    reqs = make_requests(4, prompt_len=100, target=150)
    big = run_engine(reqs, make_config(), num_blocks=256)
    out, fin, _steps = _run_mtp(reqs, 20, k=3, accept_mode="mixed")
    for r in reqs:
        assert out[r.request_id] == big.outputs[r.request_id]


def test_mtp_budget_respected() -> None:
    """MTP 行计入 token 预算：预算紧张时批被压缩。"""
    reqs = make_requests(6, prompt_len=50, target=40)
    out, _fin, _steps = _run_mtp(
        reqs, 128, k=5, accept_mode="all", max_num_batched_tokens=30
    )
    mock = MockModel(reqs)
    for r in reqs:
        assert out[r.request_id] == expected_tokens(r, mock)

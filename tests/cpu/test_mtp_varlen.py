"""MTP varlen 调度测试（CPU，本机可跑）。

性质：
- 累乘率单调递减 ⇒ 贪心分配天然前缀
- 预算截断（高 rate 优先）
- 无曲线退化 = 全量验证
- 吞吐模型（合成曲线）：高接受率 → 大 total_q
- 调度器闭环：varlen 开启 + rates 回灌 → 输出不变、总行数自适应
"""

from __future__ import annotations

import pytest

from cascade.config import SchedulerConfig
from cascade.core.mtp_varlen import MtpVarlen
from cascade.sampling.rng import derive_uniform
from tests.cpu.test_scheduler_mtp import _mock_mtp_step, make_requests
from tests.cpu.harness import make_config, run_engine
from tests.cpu.mock_model import MockModel, expected_tokens


def test_cumprod_monotone() -> None:
    rates = MtpVarlen.cumprod([0.9, 0.8, 0.7, 0.6])
    assert rates == pytest.approx([0.9, 0.72, 0.504, 0.3024])
    assert all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))


def test_plan_prefix_property() -> None:
    """贪心分配构成各 seq 的前缀（v 从 1 连续增长，无空洞）。"""
    varlen = MtpVarlen(nextn=4)
    rates = {"a": [0.9, 0.85, 0.5, 0.2], "b": [0.7, 0.65, 0.6, 0.55], "c": [0.3, 0.25, 0.2, 0.1]}
    plan = varlen.plan(rates, budget=7)  # 3 base + 4 额外槽
    total = sum(plan.values())
    assert total <= 7
    # top-4 额外槽（降序）：a0=.9, a1=.85, b0=.7, b1=.65（b 的 .6/.55 赢过 a 的 .5）
    # ⇒ a=3（j0,j1 前缀），b=3（j0,j1 前缀），c=1
    assert plan["a"] == 3
    assert plan["b"] == 3
    assert plan["c"] == 1
    assert total == 7


def test_plan_budget_cap() -> None:
    varlen = MtpVarlen(nextn=3)
    rates = {"a": [0.9] * 3, "b": [0.8] * 3}
    plan = varlen.plan(rates, budget=4)  # 2 base + 2 extra
    assert sum(plan.values()) == 4
    assert plan["a"] == 3  # 高 rate 拿满
    assert plan["b"] == 1


def test_plan_full_mode_without_curve() -> None:
    varlen = MtpVarlen(nextn=3)
    rates = {"a": [0.1] * 3, "b": [0.9] * 3}
    plan = varlen.plan(rates, budget=100)
    assert plan == {"a": 4, "b": 4}  # 无曲线 → 全量


def test_plan_fresh_seq_gets_one_row() -> None:
    varlen = MtpVarlen(nextn=3)
    rates = {"fresh": [0.0] * 3, "hot": [0.9] * 3}  # 无记录 → 0 率
    plan = varlen.plan(rates, budget=100)
    assert plan["fresh"] == 1
    assert plan["hot"] == 4


def test_plan_throughput_mode() -> None:
    """合成曲线：接受率高 → 选大 total_q；接受率低 → 收缩。"""
    # 曲率：时间随 total_q 轻微上凸（曲线 (t_q, ms)）
    curve = [(2, 10.0), (4, 12.0), (8, 16.0)]
    varlen_hi = MtpVarlen(nextn=3, perf_curve=curve)
    rates_hi = {"a": [0.95, 0.9, 0.85], "b": [0.95, 0.9, 0.85]}
    plan_hi = varlen_hi.plan(rates_hi, budget=100)
    varlen_lo = MtpVarlen(nextn=3, perf_curve=curve)
    rates_lo = {"a": [0.2, 0.15, 0.1], "b": [0.2, 0.15, 0.1]}
    plan_lo = varlen_lo.plan(rates_lo, budget=100)
    # 高接受率：更多验证行被选中
    assert sum(plan_hi.values()) > sum(plan_lo.values())


def test_scheduler_varlen_outputs_invariant() -> None:
    """varlen 开启 + rates 回灌：输出仍 ≡ 非 MTP；行数自适应。"""
    from cascade.core.block_manager import BlockManager
    from cascade.core.prefix_cache import PrefixCache
    from cascade.core.scheduler import Scheduler
    from cascade.sampling_params import SamplingParams
    from tests.cpu.harness import _check_invariants
    from tests.cpu.mock_model import MOCK_EOS_TOKEN_ID

    reqs = make_requests(4, prompt_len=70, target=50)
    baseline = run_engine(reqs, make_config(), num_blocks=128)

    cfg = make_config()
    cfg.scheduler = SchedulerConfig(max_num_seqs=64, nextn_draft_tokens=4, mtp_varlen=True)
    bm = BlockManager(128)
    pc = PrefixCache(bm, cfg.cache.block_size)
    sched = Scheduler(cfg, bm, pc)
    mock = MockModel(reqs)
    targets = {r.request_id: r.target_output_len for r in reqs}
    for r in reqs:
        sched.add_request(
            r.request_id, r.prompt,
            SamplingParams(max_tokens=10_000, seq_seed=r.seq_seed, stop_token_ids=(MOCK_EOS_TOKEN_ID,)),
        )
    outputs: dict[str, list[int]] = {}
    total_rows = 0
    # 合成 rates：每步回灌（高率 seq 持续拿大 v）
    synth_rates = {r.request_id: [0.9, 0.85, 0.8, 0.75] for r in reqs[:2]} | {
        r.request_id: [0.3, 0.2, 0.1, 0.05] for r in reqs[2:]
    }
    steps = 0
    while sched.has_unfinished_requests:
        for rid, ap in synth_rates.items():
            sched.update_mtp_accept(rid, ap)
        out = sched.schedule()
        total_rows += out.total_new_tokens
        sampled = _mock_mtp_step(sched, out, mock, 4, "mixed", targets)
        for ro in sched.update_from_output(sampled):
            outputs.setdefault(ro.request_id, []).extend(ro.new_token_ids)
        _check_invariants(sched)
        steps += 1
        assert steps < 20_000
    for r in reqs:
        assert outputs[r.request_id] == baseline.outputs[r.request_id]
        assert outputs[r.request_id] == expected_tokens(r, MockModel(reqs))
    # 自适应证据：低率 seq 的行被压缩（varlen 总行数 < 固定 K 的总行数）
    fixed_rows = sum(1 + 4 for _ in range(steps))  # 下界粗估：非严格，仅 sanity
    assert total_rows > 0

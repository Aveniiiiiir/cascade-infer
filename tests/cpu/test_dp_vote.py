"""DP 步型投票测试（CPU，本机可跑）。"""

from __future__ import annotations

from cascade.distributed.dp_vote import DPStepVoter


def test_all_ranks_have_work() -> None:
    v = DPStepVoter(all_interval=70, any_interval=100)
    assert v.decide(step=5, group_has_prefill=[True, True, True]) == "prefill"
    assert v.decide(step=70, group_has_prefill=[True, True]) == "prefill"


def test_any_point_anti_starvation() -> None:
    v = DPStepVoter(all_interval=70, any_interval=100)
    # ANY 点（step 100）：少数 rank 有活 → 集体 prefill（防饿死）
    assert v.decide(step=100, group_has_prefill=[True, False, False]) == "prefill"


def test_all_point_requires_unanimity() -> None:
    v = DPStepVoter(all_interval=70, any_interval=100)
    # ALL 点（step 70，非 ANY 点）：部分有活 → 集体 decode
    assert v.decide(step=70, group_has_prefill=[True, False]) == "decode"
    # step 0 同时是两个点（0 % 任何 == 0）→ ANY 语义优先（任一即 prefill）
    assert v.decide(step=0, group_has_prefill=[True, False]) == "prefill"


def test_normal_step_prefill_when_any_work() -> None:
    v = DPStepVoter()
    assert v.decide(step=42, group_has_prefill=[False, True]) == "prefill"
    assert v.decide(step=42, group_has_prefill=[False, False]) == "decode"


def test_empty_group_falls_back_decode() -> None:
    v = DPStepVoter()
    assert v.decide(step=100, group_has_prefill=[False, False]) == "decode"


def test_exchange_flags_single_rank() -> None:
    assert DPStepVoter.exchange_flags(True) == [True]
    assert DPStepVoter.exchange_flags(False) == [False]

"""SamplingParams 入参校验测试（审查 #9/#14 配套）。"""

from __future__ import annotations

import pytest

from cascade.sampling_params import SamplingParams


def test_top_k_validation() -> None:
    assert SamplingParams(top_k=-1).top_k == -1
    assert SamplingParams(top_k=50).top_k == 50
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=-2)
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=2**31)


def test_stop_token_seqs_normalization() -> None:
    sp = SamplingParams(stop_token_seqs=[[5, 7], [9]])
    assert sp.stop_token_seqs == ((5, 7), (9,))
    assert sp.snapshot().stop_token_seqs == ((5, 7), (9,))
    # None / 空列表 / 含空序列 → ()
    assert SamplingParams().stop_token_seqs == ()
    assert SamplingParams(stop_token_seqs=[]).stop_token_seqs == ()
    assert SamplingParams(stop_token_seqs=[[], [1]]).stop_token_seqs == ((1,),)


def test_bad_params_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError, match="seq_seed"):
        SamplingParams(seq_seed=-1)
    with pytest.raises(ValueError, match="seq_seed"):
        SamplingParams(seq_seed=2**64)

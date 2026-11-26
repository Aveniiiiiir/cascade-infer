"""NGramDraftModel 匹配逻辑（纯 Python，无 torch）。"""

from __future__ import annotations

from cascade.sampling.ngram_draft import NGramDraftModel


def test_repeated_pattern_hits() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=4, max_draft=3)
    # "1 2 3 1 2 3" 的后缀 1 2 3 在位置 0 出现过 → 草案 = 1 2 3
    toks, probs = m.lookup([1, 2, 3, 1, 2, 3], 3)
    assert toks == [1, 2, 3] and probs == [1.0, 1.0, 1.0]


def test_partial_continuation_padded_with_sentinel() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=4, max_draft=4)
    # 后缀 "7 8" 在位置 1 出现，其后是 9 7 8（周期重复）→ 3 个真实草案，
    # 第 4 个位置为 -1 哨兵
    toks, probs = m.lookup([5, 7, 8, 9, 7, 8], 4)
    assert toks == [9, 7, 8, -1]
    assert probs == [1.0, 1.0, 1.0, 0.0]


def test_no_match_all_sentinel() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=4, max_draft=3)
    toks, probs = m.lookup([10, 20, 30, 40, 50], 3)
    assert toks == [-1, -1, -1] and probs == [0.0, 0.0, 0.0]


def test_short_context() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=4, max_draft=3)
    assert m.lookup([1], 3) == ([-1, -1, -1], [0.0, 0.0, 0.0])
    assert m.lookup([1, 2], 3) == ([-1, -1, -1], [0.0, 0.0, 0.0])


def test_exact_suffix_occurrence_not_self_matched() -> None:
    """后缀自身那次出现不能作为匹配（否则草案恒为空）。"""
    m = NGramDraftModel(min_gram=2, max_gram=4, max_draft=2)
    # 只出现一次 "4 5"（就在末尾）→ 无候选
    assert m.lookup([1, 2, 3, 4, 5], 2) == ([-1, -1], [0.0, 0.0])


def test_longest_gram_preferred() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=3, max_draft=2)
    # 3-gram "1 2 3" 出现在位置 0，2-gram "2 3" 出现在位置 1；应选 3-gram
    # → 其后 token = 4（位置 3）
    toks, _ = m.lookup([1, 2, 3, 4, 9, 1, 2, 3], 2)
    assert toks[0] == 4


def test_truncated_to_k() -> None:
    m = NGramDraftModel(min_gram=2, max_gram=2, max_draft=5)
    toks, probs = m.lookup([1, 2, 1, 2, 1, 2], 2)
    assert len(toks) == 2 and len(probs) == 2


def test_large_context_bytes_roundtrip() -> None:
    """uint32 打包对大 token id（>65535）不失真。"""
    m = NGramDraftModel(min_gram=2, max_gram=2, max_draft=2)
    toks, _ = m.lookup([120000, 120001, 120002, 120000, 120001], 2)
    assert toks[0] == 120002
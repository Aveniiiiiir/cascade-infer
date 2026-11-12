"""SO 层单元测试（纯 Python，本机可跑）：stub 文法 / 异步 builder / 失败隔离。"""

from __future__ import annotations

import time

import pytest

from cascade.structured_output.builder import SOMatchersBuilder
from cascade.structured_output.compiler import GrammarCompiler, has_xgrammar
from cascade.structured_output.matcher import (
    SO_STATUS_INACTIVE,
    StubGrammarBackend,
    SOMatcher,
)

VOCAB = 128


def test_compiler_dispatch() -> None:
    c = GrammarCompiler()
    g = c.compile({"schema_type": "regex", "schema": "5,7,2"})
    assert g.schema_type == "regex" and g.schema == "5,7,2"
    # 缓存命中
    assert c.compile({"schema_type": "regex", "schema": "5,7,2"}) is g
    # 非法输入
    with pytest.raises(ValueError):
        c.compile({"schema_type": "xml", "schema": "x"})


def test_compiler_json_without_xgrammar_raises() -> None:
    if has_xgrammar():
        pytest.skip("本机装了 xgrammar（走真后端，错误路径不适用）")
    c = GrammarCompiler()
    with pytest.raises(RuntimeError, match="xgrammar"):
        c.compile({"schema_type": "json", "schema": {"type": "object"}})


def test_stub_backend_mask_semantics() -> None:
    m = SOMatcher(StubGrammarBackend("5,7,2"), VOCAB)
    words = [0] * m.num_words
    m.fill_mask_into(words)
    assert words[5 // 32] == 1 << (5 % 32)  # 只有 token 5 合法
    assert sum(bin(w).count("1") for w in words) == 1
    # 接受 5 → 下一步只允许 7
    m.accept_token(5)
    words = [0] * m.num_words
    m.fill_mask_into(words)
    assert words[7 // 32] == 1 << (7 % 32)
    # 回滚 1 步 → 回到只允许 5
    m.rollback(1)
    words = [0] * m.num_words
    m.fill_mask_into(words)
    assert words[5 // 32] == 1 << (5 % 32)


def test_stub_backend_terminated_full_open() -> None:
    m = SOMatcher(StubGrammarBackend("5"), VOCAB)
    assert not m.terminated
    m.accept_token(5)
    assert m.terminated
    words = [0] * m.num_words
    m.fill_mask_into(words)  # terminated → 全开（-1 = 32 个 1）
    assert all(w == -1 for w in words)


def test_builder_async_submit_pump() -> None:
    b = SOMatchersBuilder()
    b.submit("r1", {"schema_type": "regex", "schema": "5,7"}, VOCAB)
    b.submit("r1", {"schema_type": "regex", "schema": "5,7"}, VOCAB)  # 幂等
    deadline = time.time() + 5.0
    results = []
    while time.time() < deadline:
        results = b.pump()
        if results:
            break
        time.sleep(0.01)
    assert len(results) == 1
    r = results[0]
    assert r.request_id == "r1" and r.matcher is not None and r.error is None
    assert r.matcher.status == SO_STATUS_INACTIVE
    b.shutdown()


def test_builder_failure_isolation() -> None:
    if has_xgrammar():
        pytest.skip("本机装了 xgrammar（json 编译会成功）")
    b = SOMatchersBuilder()
    b.submit("bad", {"schema_type": "json", "schema": {"type": "object"}}, VOCAB)
    b.submit("good", {"schema_type": "regex", "schema": "3"}, VOCAB)
    deadline = time.time() + 5.0
    results: list = []
    while time.time() < deadline:
        results += b.pump()  # 累积（早轮收割的结果不能丢弃）
        if len(results) >= 2:
            break
        time.sleep(0.01)
    by_id = {r.request_id: r for r in results}
    assert set(by_id) >= {"bad", "good"}
    assert by_id["bad"].matcher is None and by_id["bad"].error is not None
    assert by_id["good"].matcher is not None
    b.shutdown()


def test_stub_rollback_batch() -> None:
    """试探性接受 + 回滚（MTP 掩码用法）。"""
    m = SOMatcher(StubGrammarBackend("1,2,3,4,5"), VOCAB)
    masks = []
    for j in range(3):
        w = [0] * m.num_words
        m.fill_mask_into(w)
        masks.append(w)
        m.accept_token(j + 1)  # 试探
    m.rollback(3)  # 全部回滚
    w = [0] * m.num_words
    m.fill_mask_into(w)
    assert w == masks[0]  # 回到初始态

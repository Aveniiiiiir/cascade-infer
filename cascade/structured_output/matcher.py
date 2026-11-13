"""SO matcher：掩码填充 / token 接受 / 回滚 的后端接口 + 两个实现。

掩码契约（与 triton_ops.apply_so_bitmask 一致）：int32 位掩码行
[ceil(vocab/32)]，第 i 词第 j 位 = 1 表示 token (i*32+j) 合法。

- XGrammarBackend：集群真后端（xgrammar.GrammarMatcher：
  fill_next_token_bitmask / accept_token / rollback / terminated）
- StubGrammarBackend：确定性逐步约束（本地引擎路径测试）——
  allowed_token(step) = f(schema)（如 "0,1,2,...,eos" 的数字序列文法）

MTP 多步掩码用法（DeepInfer matcher.py:366-376 语义）：
    for j in range(K):
        fill_mask_into(row_j)         # 当前状态的下一步掩码
        accept_token(draft_j)          # 试探性接受（填下一行掩码用）
    rollback(K)                        # 回滚全部试探（真实状态未动）
    ...验证产出 committed 后再真实 accept...
"""

from __future__ import annotations

from typing import Sequence

SO_STATUS_INACTIVE = 0
SO_STATUS_ACTIVATING = 1
SO_STATUS_ACTIVE = 2
SO_STATUS_FAILED = -1


class GrammarBackend:
    """后端最小接口（fill / accept / rollback / terminated）。"""

    def fill_mask_into(self, words: Sequence[int], vocab_size: int) -> None:
        raise NotImplementedError

    def accept_token(self, token_id: int) -> None:
        raise NotImplementedError

    def rollback(self, k: int) -> None:
        raise NotImplementedError

    @property
    def terminated(self) -> bool:
        raise NotImplementedError


class XGrammarBackend(GrammarBackend):
    """xgrammar.GrammarMatcher 包装（集群）。掩码行宽 = ceil(vocab/32)。"""

    __slots__ = ("matcher", "vocab_size", "_buf")

    def __init__(self, compiled_grammar, vocab_size: int, tokenizer_info=None) -> None:
        import xgrammar  # 局部导入：仅集群路径

        if tokenizer_info is not None:
            self.matcher = xgrammar.GrammarMatcher(
                compiled_grammar, tokenizer_info=tokenizer_info
            )
        else:
            self.matcher = xgrammar.GrammarMatcher(compiled_grammar)
        self.vocab_size = vocab_size
        self._buf = None  # 掩码缓冲复用（审查 #22：免每行每步分配）

    def fill_mask_into(self, words: Sequence[int], vocab_size: int) -> None:
        import torch

        if self._buf is None or self._buf.shape[1] != len(words):
            self._buf = torch.zeros(1, len(words), dtype=torch.int32)
        else:
            self._buf.zero_()
        # xgrammar 填充 torch int32 [1, words*32 对齐]；落到调用方的行缓冲
        self.matcher.fill_next_token_bitmask(self._buf, 0)
        w = self._buf[0].tolist()
        for i in range(len(words)):
            words[i] = int(w[i])

    def accept_token(self, token_id: int) -> None:
        self.matcher.accept_token(token_id)

    def rollback(self, k: int) -> None:
        self.matcher.rollback(k)

    @property
    def terminated(self) -> bool:
        return self.matcher.terminated()


class StubGrammarBackend(GrammarBackend):
    """确定性 stub：``schema`` = 逗号分隔 token 序列（如 "5,7,9,2"）。

    语义：第 step 步只允许 schema[step]；接受 schema 末位后 terminated。
    回滚按步数退回。本地引擎路径测试用（掩码/接受/回滚/终止闭环）。
    """

    __slots__ = ("_allowed", "_step")

    def __init__(self, schema: str) -> None:
        self._allowed = [int(x) for x in schema.split(",") if x.strip() != ""]
        self._step = 0

    def fill_mask_into(self, words: Sequence[int], vocab_size: int) -> None:
        for i in range(len(words)):
            words[i] = 0
        tok = self._allowed[self._step] if self._step < len(self._allowed) else -1
        if tok < 0:
            for i in range(len(words)):  # terminated：全开（EOS 后续写语义）
                words[i] = -1
            return
        words[tok // 32] |= 1 << (tok % 32)

    def accept_token(self, token_id: int) -> None:
        self._step += 1  # 宽松接受（约束由掩码施加；接受只推进状态）

    def rollback(self, k: int) -> None:
        self._step = max(0, self._step - k)

    @property
    def terminated(self) -> bool:
        return self._step >= len(self._allowed)


class SOMatcher:
    """引擎侧句柄：后端 + so_status 状态机 + 失败标记。"""

    __slots__ = ("backend", "vocab_size", "num_words", "status")

    def __init__(self, backend: GrammarBackend, vocab_size: int) -> None:
        self.backend = backend
        self.vocab_size = vocab_size
        self.num_words = (vocab_size + 31) // 32
        self.status = SO_STATUS_INACTIVE

    # ------------------------------------------------------------------ ops

    def fill_mask_into(self, words: Sequence[int]) -> None:
        """下一步合法 token 位掩码 → 写入 words（长度 num_words）。"""
        self.backend.fill_mask_into(words, self.vocab_size)

    def accept_token(self, token_id: int) -> None:
        self.backend.accept_token(token_id)

    def rollback(self, k: int) -> None:
        self.backend.rollback(k)

    @property
    def terminated(self) -> bool:
        return self.backend.terminated

    def fill_ones(self, words: Sequence[int]) -> None:
        """无约束行（非 SO 行 / terminated 全开）：全 1。"""
        for i in range(len(words)):
            words[i] = -1


__all__ = [
    "GrammarBackend",
    "XGrammarBackend",
    "StubGrammarBackend",
    "SOMatcher",
    "SO_STATUS_INACTIVE",
    "SO_STATUS_ACTIVATING",
    "SO_STATUS_ACTIVE",
    "SO_STATUS_FAILED",
]

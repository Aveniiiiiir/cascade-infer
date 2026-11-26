"""无 draft 模型的 n-gram 投机解码（prompt-lookup / suffix decoding 同族）。

草案来源 = 序列自身历史的**最长后缀匹配**：在 ``prompt‖已生成`` 的 token 串
上取最后 n 个 token（n 从 ``max_gram`` 递减到 ``min_gram``），用 ``bytes.rfind``
（C 速度）找它此前最近一次出现，把其后的最多 k 个 token 作为草案。

设计要点：
- **one-hot 草案分布**：匹配位置的 draft 概率 = 1.0，未命中位置 = -1 哨兵
  （mtp_verify 的 fresh 分支强制拒绝）→ 复用 MTP 的拒绝采样验证，输出分布
  保持（贪心下 MTP ≡ 非 MTP 逐 token 一致，见 sampling/mtp_verify.py）。
- **不做 [K, V] 全量 logprobs**：``sparse_draft = True`` 时引擎走
  ``mtp_verify(draft_token_probs=...)`` 稀疏路径（N-gram 的 V=128k，
  全量分配每步几十 MB，纯浪费）。
- **无状态**：每次调用由引擎传入完整上下文（prompt‖已生成‖本步已提交），
  不维护 per-request 索引——被抢占/重算/abort 后无需清理，杜绝状态腐蚀。
- 复杂度：每次查找 O(ctx_len × gram) 次字节比较（memchr 级），4k 上下文
  × 32 seq 实测可忽略；不做 Python 逐 token 扫描。

适用面：重复/引用/代码编辑类输出（acceptance 高），通用文本接近 0 命中时
草案全为哨兵，验证成本 = 固定 K 行的前向（与 MTP 固定 K 相同）。
"""

from __future__ import annotations

import math
from array import array

#: token id 必须能放进 uint32（DeepSeek 家族 vocab ≈ 128k，恒成立）
_PACK = "I"
_PACK_SIZE = 4


class NGramDraftModel:
    """DraftModel 协议的稀疏实现（引擎在 ``draft_model`` 位置注入）。"""

    #: 引擎据此走稀疏验证（不分配 [K, V] logprobs）
    sparse_draft = True

    __slots__ = ("min_gram", "max_gram", "max_draft")

    def __init__(self, min_gram: int = 2, max_gram: int = 4, max_draft: int = 5) -> None:
        """Args:
            min_gram: 最短匹配 gram（太小会命中噪声，2 是 prompt-lookup 常用值）
            max_gram: 最长匹配 gram（越大越准，命中率越低）
            max_draft: 单次最多草案 token 数（引擎调用时再按 k 截断）
        """
        assert 1 <= min_gram <= max_gram
        assert max_draft >= 1
        self.min_gram = min_gram
        self.max_gram = max_gram
        self.max_draft = max_draft

    # ------------------------------------------------------------------ DraftModel 协议

    def draft_tokens(
        self,
        last_hidden: object,
        last_token: object,
        start_positions: object,
        k: int,
        *,
        context_tokens: list[list[int]] | None = None,
        request_ids: list[str] | None = None,
    ) -> tuple[list[list[int]], None, list[list[float]]]:
        """→ (tokens [n][k]、None（无全量 logprobs）、probs [n][k])。

        tokens[i][j] = -1 表示该位置无草案（fresh 哨兵，验证时强制拒绝）。
        probs[i][j] 为该草案 token 的概率（命中 = 1.0，哨兵 = 0.0）。
        """
        if context_tokens is None:
            raise ValueError("NGramDraftModel 需要 context_tokens（引擎须传序列历史）")
        n = len(context_tokens)
        tokens: list[list[int]] = []
        probs: list[list[float]] = []
        for ctx in context_tokens:
            row, prow = self.lookup(ctx, k)
            # 引擎的验证长度 K 可能大于本模型的 max_draft：尾部补哨兵，
            # 保证 len(tokens) == k（否则 _verify_mtp_seq 视为无草案）
            if len(row) < k:
                row = list(row) + [-1] * (k - len(row))
                prow = list(prow) + [0.0] * (k - len(prow))
            tokens.append(row)
            probs.append(prow)
        assert len(tokens) == n and len(probs) == n
        return tokens, None, probs

    # ------------------------------------------------------------------ 匹配核心

    def lookup(self, context: list[int], k: int) -> tuple[list[int], list[float]]:
        """最长后缀匹配 → (k 个草案 token（-1 填充）, 对应概率)。"""
        k = min(k, self.max_draft)
        pad_tokens = [-1] * k
        pad_probs = [0.0] * k
        if k == 0 or len(context) < self.min_gram + 1:
            return pad_tokens, pad_probs
        ctx = context
        total = len(ctx)
        buf = array(_PACK, ctx).tobytes()  # 小端/本地字节序一致即可（自比较）
        max_n = min(self.max_gram, total - 1)
        for n in range(max_n, self.min_gram - 1, -1):
            # 匹配必须整体落在 [0, total-1) 内：排除「正好是查询后缀自身」的
            # 那一次出现（否则草案 = 空），并保证至少还有 1 个后续 token。
            end_bytes = (total - 1) * _PACK_SIZE
            if end_bytes < n * _PACK_SIZE:
                continue
            sub = array(_PACK, ctx[-n:]).tobytes()
            pos = buf.rfind(sub, 0, end_bytes)
            if pos < 0:
                continue
            s = pos // _PACK_SIZE
            cont = ctx[s + n : s + n + k]
            if not cont:
                continue
            out_tokens = list(cont)[:k]
            out_probs = [1.0] * len(out_tokens)
            out_tokens += [-1] * (k - len(out_tokens))
            out_probs += [0.0] * (k - len(out_probs))
            return out_tokens, out_probs
        return pad_tokens, pad_probs

    # ------------------------------------------------------------------ 稀疏验证辅助

    @staticmethod
    def draft_log_prob(p: float) -> float:
        """one-hot 草案的 log p（0.0 或 -inf；供 mtp_verify 稀疏路径复用）。"""
        return 0.0 if p >= 1.0 else (math.log(p) if p > 0.0 else float("-inf"))


__all__ = ["NGramDraftModel"]
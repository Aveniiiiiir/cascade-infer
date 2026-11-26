"""MTP 验证：采样式拒绝采样（DeepInfer fused_sample.py:227-248 语义）。

行对齐（0 索引草案 d[0..K-1]）：
    输入 [t_last, d_0, ..., d_{K-1}] → 目标 logits L[0..K]
    d[j] 的验证：目标分布 q_j = softmax(L[j])，draft 分布 p[j]，
    接受 ⇔ U_j < exp(clamp(q_lp[j][d_j] − p_lp[j][d_j], max=0))
    拒绝 → 本位置 token = 逆 CDF(relu(q_j − α·p[j]), u_j)，停止接受
    全接受 → bonus token = 逆 CDF(q_K, u_K)
    α = exp(−DRAFT_PROB_PENALTY)，输出分布偏差有界（DeepInfer common.py
    的证明：(1+eps)·q ≥ A ≥ q − eps/(1+eps)）

关键性质：
- K = 0（无草案；varlen 的 fresh 行 / -1 哨兵）退化为普通确定性采样——
  残差 = relu(q − 0) = q
- 贪心（目标与草案均贪心）⇒ 接受 ⇔ token 相同；拒绝行的残差 argmax =
  目标 argmax（p 在错 token 上 one-hot）⇒ **MTP 输出与非 MTP 逐 token
  一致**（引擎级验收的根基）
- 理论接受率 = Σ min(p, q)（合成分布测试的 oracle）
- u_j 取 case=0 流于绝对位置 P+j+1 —— 与非 MTP 采样同一随机流
  （fallback 位 bit 级兼容）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cascade.model_executor.triton_ops.common import DRAFT_PROB_PENALTY

_ALPHA = torch.exp(torch.tensor(-DRAFT_PROB_PENALTY)).item()


@dataclass(slots=True)
class MtpVerifyResult:
    #: 每个已处理草案位置的 token（接受 = d_j；拒绝 = 残差重采样），
    #: 长度 = 已处理草案数（中途拒绝则截断）
    row_tokens: list[int]
    #: bonus token（全接受时追加；K=0 时即唯一产出；中途拒绝 = -1）
    bonus_token: int
    accepted: list[bool]  # [已处理数]
    accept_probs: list[float]  # [已处理数]（真实接受概率的在线估计）
    committed: list[int]  # 最终提交的 token 序列（含截断）
    n_accepted_drafts: int


def mtp_verify(
    target_logprobs: torch.Tensor,  # [K+1, V] fp32（log_softmax 后）
    draft_logprobs: torch.Tensor | None,  # [K, V] fp32（K=0 时形状 [0, V]；稀疏路径 None）
    draft_tokens: torch.Tensor,  # [K] int（-1 = fresh 哨兵 → 强制拒绝）
    accept_u: torch.Tensor,  # [K] fp32（verify 流均匀数）
    sample_u: torch.Tensor,  # [K+1] fp32（case=0 流；[j] = 位置 P+j+1）
    draft_token_probs: torch.Tensor | None = None,  # [K] fp32：稀疏（one-hot）草案分布
) -> MtpVerifyResult:
    """单请求的 MTP 验证（torch 参考实现；fused Triton kernel 后置）。

    ``draft_token_probs`` 非 None 时走**稀疏草案**路径（n-gram/prompt-lookup：
    草案分布是单点 p[d]，无需 [K, V] 全量分配）：
        ratio = exp(min(q_lp[d] − log p_d, 0))
        拒绝残差 = relu(q − α·p·e_d)
        接受概率 = min(q[d], p_d)（该式即 Σ_x min(p,q) 在单点分布上的精确值，
        比全量路径的 1−Σrelu 估计更准，varlen 接受率回灌因此不虚高）
    """
    K1, V = target_logprobs.shape
    K = K1 - 1
    if draft_logprobs is not None:
        assert draft_logprobs.shape == (K, V), (draft_logprobs.shape, (K, V))
        assert draft_token_probs is None, "稀疏/全量草案分布二选一"
    else:
        assert draft_token_probs is not None and draft_token_probs.shape == (K,), (
            "稀疏路径必须提供 draft_token_probs [K]"
        )
    assert draft_tokens.shape == (K,)
    assert accept_u.shape == (K,) and sample_u.shape == (K + 1,)
    q = target_logprobs.exp()  # [K+1, V] 概率（未归一亦可，逆 CDF 自适应）

    row_tokens: list[int] = []
    accepted: list[bool] = []
    accept_probs: list[float] = []
    committed: list[int] = []
    n_accepted = 0

    for j in range(K):
        d = int(draft_tokens[j])
        q_lp = target_logprobs[j]
        if draft_logprobs is None:
            p_d = float(draft_token_probs[j])  # type: ignore[index]
            p_lp = None
            valid_draft = d >= 0 and p_d > 0.0
        else:
            p_d = 0.0
            p_lp = draft_logprobs[j]
            valid_draft = d >= 0
        if not valid_draft:
            # fresh 哨兵（varlen 的 v=0 行 / n-gram 未命中）：强制拒绝，残差 = q
            token = _inv_cdf(q[j], float(sample_u[j]))
            row_tokens.append(token)
            accepted.append(False)
            accept_probs.append(1.0)
            committed.append(token)
            return MtpVerifyResult(row_tokens, -1, accepted, accept_probs, committed, 0)
        if p_lp is not None:
            ratio = torch.exp(torch.clamp(q_lp[d] - p_lp[d], max=0.0))
        else:
            # 稀疏：p[d] = p_d（其余列 0）→ log p_d；零概率走上面的 fresh 分支
            ratio = math.exp(min(float(q_lp[d]) - math.log(p_d), 0.0))
        if float(accept_u[j]) < float(ratio):
            row_tokens.append(d)
            accepted.append(True)
            accept_probs.append(
                _accept_prob(q[j], p_lp) if p_lp is not None else _accept_prob_sparse(q[j], d, p_d)
            )
            committed.append(d)
            n_accepted += 1
        else:
            if p_lp is not None:
                residual = torch.relu(q[j] - _ALPHA * p_lp.exp())
            else:
                # relu(q − α·p_d·e_d)：只需在 d 列做减法
                residual = q[j].clone()
                residual[d] = torch.clamp(residual[d] - _ALPHA * p_d, min=0.0)
            # 残差全零（draft 分布在所有列都 ≥ target/α）时回退到 target
            # 分布 q 本身——拒绝采样的偏差界允许此退化（DeepInfer 同语义），
            # 否则 _inv_cdf 对全零行报错杀死请求
            if float(residual.sum()) <= 0.0:
                residual = q[j]
            token = _inv_cdf(residual, float(sample_u[j]))
            row_tokens.append(token)
            accepted.append(False)
            accept_probs.append(
                _accept_prob(q[j], p_lp) if p_lp is not None else _accept_prob_sparse(q[j], d, p_d)
            )
            committed.append(token)
            return MtpVerifyResult(row_tokens, -1, accepted, accept_probs, committed, n_accepted)
    # 全接受 → bonus（K=0 时直接到这里：普通确定性采样）
    bonus = _inv_cdf(q[K], float(sample_u[K]))
    return MtpVerifyResult(
        row_tokens=row_tokens,
        bonus_token=bonus,
        accepted=accepted,
        accept_probs=accept_probs,
        committed=[*[int(draft_tokens[j]) for j in range(K)], bonus],
        n_accepted_drafts=K,
    )


def _inv_cdf(probs: torch.Tensor, u: float) -> int:
    """逆 CDF（零概率列永不命中；未归一化行自适应 total）。"""
    cdf = torch.cumsum(probs.double(), dim=0)
    target = u * float(cdf[-1])
    mask = cdf > target
    if bool(mask.any()):
        # Bool argmax 在 torch 未实现（"argmax_cpu" not implemented for 'Bool'）
        return int(mask.long().argmax().item())
    for i in range(probs.numel() - 1, -1, -1):
        if float(probs[i]) > 0:
            return i
    raise ValueError("_inv_cdf: 全零概率行")


def _accept_prob(q: torch.Tensor, p_lp: torch.Tensor) -> float:
    """accept_probs = clamp(1 − Σ residual, 0)：真实接受概率的估计
    （DeepInfer 用它做置信度头的在线校准监控）。"""
    residual = torch.relu(q - _ALPHA * p_lp.exp())
    return float(torch.clamp(1.0 - residual.sum(), min=0.0).item())


def _accept_prob_sparse(q: torch.Tensor, d: int, p_d: float) -> float:
    """单点草案分布的精确接受概率 Σ_x min(p(x), q(x)) = min(p_d, q[d])。"""
    return float(min(max(float(q[d]), 0.0), max(p_d, 0.0)))


__all__ = ["mtp_verify", "MtpVerifyResult", "DRAFT_PROB_PENALTY"]

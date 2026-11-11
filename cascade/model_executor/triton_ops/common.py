"""算子层共享常量（对应 DeepInfer tilelang_ops/common.py）。"""

from __future__ import annotations

#: MTP 拒绝采样残差分布的偏差控制常数。
#: residual = relu(q − α·p)，α = exp(−DRAFT_PROB_PENALTY)。
#: DeepInfer common.py:3-53 给出完整偏差界证明：输出分布满足
#: (1+eps)·q ≥ A ≥ q − eps/(1+eps)。
DRAFT_PROB_PENALTY = 5e-5

#: draft/nextn 独立随机流偏移（DeepInfer fused_sample.py:15）
MTP_SEED_OFFSET = 1000000007

#: 用途流编码（DeepInfer random_sampling.py:78 的 `* 3 + case`）
CASE_SAMPLE = 0  # 普通采样
CASE_NEXTN = 1  # draft/nextn 预测
CASE_VERIFY = 2  # MTP 验证（拒绝采样）

__all__ = [
    "DRAFT_PROB_PENALTY",
    "MTP_SEED_OFFSET",
    "CASE_SAMPLE",
    "CASE_NEXTN",
    "CASE_VERIFY",
]

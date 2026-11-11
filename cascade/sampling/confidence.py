"""MTP 置信度估计与在线校准。

V3 的 nextn 层没有 DFlash 的 confidence head（V4.1 专属）——Cascade 用
draft top-1 概率做无参数代理，用验证阶段产出的 accept_probs（真实接受
概率估计，DeepInfer fused_sample 的第 6 输出）按 32 桶 EMA 在线校准：

    conf[j] = calibrated(p_top1[j])
    calibrated(x) = EMA[quantize(x)] if 桶已见 else x
    EMA[桶] ← β·EMA[桶] + (1−β)·accept_prob

理论依据：p 与 q 一致时 token 接受概率 = Σ min(p, q)，其主项是
p_top1（无需训练的接受率代理）；系统性偏差（draft 过自信/欠自信）
由 EMA 校准吸收 —— 与 DeepInfer 用 accept_probs 校准 confidence head
的闭环同构。
"""

from __future__ import annotations


class ConfidenceCalibrator:
    __slots__ = ("num_buckets", "beta", "_ema", "_seen")

    def __init__(self, num_buckets: int = 32, beta: float = 0.99) -> None:
        assert num_buckets > 0 and 0.0 < beta < 1.0
        self.num_buckets = num_buckets
        self.beta = beta
        self._ema: list[float] = [0.0] * num_buckets
        self._seen: list[bool] = [False] * num_buckets

    def quantize(self, p_top1: float) -> int:
        """p_top1 ∈ [0,1] → 桶索引（线性等分）。"""
        p = min(max(float(p_top1), 0.0), 1.0)
        idx = int(p * self.num_buckets)
        return min(idx, self.num_buckets - 1)

    def calibrate(self, p_top1: float) -> float:
        """校准后的置信度；未见的桶退回原始 p_top1（冷启动）。"""
        b = self.quantize(p_top1)
        if self._seen[b]:
            return self._ema[b]
        return min(max(float(p_top1), 0.0), 1.0)

    def update(self, p_top1: float, accept_prob: float) -> None:
        """在线校准：accept_prob 为该 draft token 的真实接受概率估计。"""
        b = self.quantize(p_top1)
        obs = min(max(float(accept_prob), 0.0), 1.0)
        if not self._seen[b]:
            self._ema[b] = obs
            self._seen[b] = True
        else:
            self._ema[b] = self.beta * self._ema[b] + (1.0 - self.beta) * obs

    def update_batch(self, p_top1s: list[float], accept_probs: list[float]) -> None:
        for p, a in zip(p_top1s, accept_probs):
            self.update(p, a)

    def calibrated_confidences(self, p_top1s: list[float]) -> list[float]:
        return [self.calibrate(p) for p in p_top1s]

    @property
    def num_seen_buckets(self) -> int:
        return sum(self._seen)


__all__ = ["ConfidenceCalibrator"]

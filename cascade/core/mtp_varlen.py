"""MTP varlen 调度（DeepInfer csrc/engine/mtp_varlen.hpp:150 语义，CPU 版）。

核心机制（报告 C）：
- 每请求维护 ``rates[j] = Π_{k≤j} confidence[k]``（累乘）——验证第
  j+1 个草案的期望增益 ≈ rate_j
- **累乘单调递减 ⇒ 贪心选 top-(total_q − n) 个 (seq, j) 槽位天然构成
  各 seq 的前缀**（rate_j ≥ rate_{j+1}，选了 j+1 必已选 j）——无需显式
  前缀约束
- total_q 预算：无性能曲线 → 全量验证（DeepInfer :170-175 退化路径）；
  有曲线（CUDA graph 逐档 timed replay 的 decode perf curve）→ 线性
  搜索最大化 ``expected_tokens / time(total_q)``

批不变性注记：varlen 与批不变互斥（DeepInfer 同此——验证长度随 batch
组合变化）。CASCADE_BATCH_INVARIANT=1 时引擎应固定 K（env 互斥由
调用方保证）。
"""

from __future__ import annotations

from bisect import bisect_left


class MtpVarlen:
    __slots__ = ("nextn", "perf_curve")

    def __init__(self, nextn: int, perf_curve: list[tuple[int, float]] | None = None) -> None:
        """perf_curve: [(total_q, ms)] 升序（Phase 4 的 profile_decode_perf
        产出；None = 全量验证退化）。"""
        assert nextn >= 0
        self.nextn = nextn
        self.perf_curve = sorted(perf_curve) if perf_curve else None

    # ------------------------------------------------------------------ plan

    def plan(self, rates: dict[str, list[float]], budget: int) -> dict[str, int]:
        """分配每 seq 的验证长度 v（1..1+K），总行数 ≤ budget。

        Args:
            rates: rid → 累乘率列表（长度 ≤ K；无记录的 seq 用默认率，
                见 plan_with_defaults）
            budget: 本步 decode 可用的总行预算（≥ seq 数）
        Returns:
            rid → v（1 + 分配的验证槽数）
        """
        assert budget >= len(rates)
        if self.nextn == 0 or not rates:
            return {rid: 1 for rid in rates}
        if self.perf_curve is None:
            # 退化：全量验证（受 budget 截断——高 rate 优先）
            total_q = min(budget, len(rates) * (1 + self.nextn))
        else:
            # 吞吐模型：线性搜索 total_q 最大化 expected/time
            n = len(rates)
            max_total = n * (1 + self.nextn)
            best_total = n
            best_tp = -1.0
            for total_q in range(n, max_total + 1):
                expected = self._expected_tokens(rates, total_q)
                t = self._interp_time(total_q)
                tp = expected / t
                if tp > best_tp:
                    best_tp = tp
                    best_total = total_q
            total_q = min(best_total, budget)
        return self._allocate(rates, total_q)

    def plan_with_defaults(self, rates: dict[str, list[float]], fresh_rate: float, budget: int) -> dict[str, int]:
        """无 rates 记录的 seq（首个 decode 步）以 fresh_rate 估计。"""
        full = {rid: rates.get(rid) or [fresh_rate] * self.nextn for rid in rates}
        # plan 需要包含所有 seq（含无记录者）
        return self.plan(full, budget)

    # ------------------------------------------------------------------ internals

    def _expected_tokens(self, rates: dict[str, list[float]], total_q: int) -> float:
        """预算 total_q 下的期望 token 数：base n + top-(total_q−n) 累乘率。"""
        n = len(rates)
        if total_q <= n:
            return float(n)
        pairs: list[float] = []
        for rl in rates.values():
            pairs.extend(rl[: self.nextn])
        pairs.sort(reverse=True)
        extra = pairs[: total_q - n]
        return float(n) + sum(extra)

    def _interp_time(self, total_q: int) -> float:
        """性能曲线线性插值（曲线外延用端点斜率）。"""
        curve = self.perf_curve
        assert curve
        if len(curve) == 1:  # 审查 #16：单点曲线无斜率可外延
            return curve[0][1]
        xs = [c[0] for c in curve]
        if total_q <= xs[0]:
            return curve[0][1]
        if total_q >= xs[-1]:
            # 端点外延：用最后一段斜率
            (x0, y0), (x1, y1) = curve[-2], curve[-1]
            slope = (y1 - y0) / max(x1 - x0, 1)
            return y1 + slope * (total_q - x1)
        i = bisect_left(xs, total_q)
        (x0, y0), (x1, y1) = curve[i - 1], curve[i]
        return y0 + (y1 - y0) * (total_q - x0) / max(x1 - x0, 1)

    def _allocate(self, rates: dict[str, list[float]], total_q: int) -> dict[str, int]:
        """贪心把 total_q 行分配为 per-seq v：top-(total_q−n) 累乘率槽位
        （累乘单调 ⇒ 天然前缀）。"""
        n = len(rates)
        pairs: list[tuple[float, str, int]] = []
        for rid, rl in rates.items():
            for j, r in enumerate(rl[: self.nextn]):
                if r > 0.0:  # 零率（无草案的 fresh seq）不占验证槽
                    pairs.append((r, rid, j))
        pairs.sort(reverse=True)
        allowed = total_q - n
        plan: dict[str, int] = {rid: 1 for rid in rates}
        for taken, (_r, rid, _j) in enumerate(pairs):
            if taken >= allowed:
                break
            plan[rid] += 1
        return plan

    @staticmethod
    def cumprod(confidences: list[float]) -> list[float]:
        """累乘率：rates[j] = Π_{k≤j} conf[k]。"""
        out: list[float] = []
        acc = 1.0
        for c in confidences:
            acc *= float(c)
            out.append(min(max(acc, 0.0), 1.0))
        return out


__all__ = ["MtpVarlen"]

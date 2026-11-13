"""DP 步型投票（DeepInfer scheduler.hpp:295-341 语义的纯函数版）。

多副本（DP 组）必须**同型步**（全组同一步做 prefill 或 decode）。
规则（报告 B 参数：ALL 点每 70 步 / ANY 点每 100 步）：
- ANY 点（step % any_interval == 0）：任一 rank 有 prefill 活动 → 集体
  prefill（防饿死——迟到请求不会被多数 decode 步无限压制）
- ALL 点（step % all_interval == 0）：全员有活才 prefill；部分有活 →
  集体 decode（此点强制全员一致，避免追赶失衡）
- 其余步：任一 rank 有活 → prefill（常规优先 prefill，与单机引擎一致）
- 全组空手 → decode（防死循环兜底）

集群接入：两轮 gloo allreduce 交换各 rank 的 has_prefill 标志
（CPU 组，避免 CUDA sync——DeepInfer 同此），本函数为决策核。
"""

from __future__ import annotations


class DPStepVoter:
    __slots__ = ("all_interval", "any_interval")

    def __init__(self, all_interval: int = 70, any_interval: int = 100) -> None:
        assert all_interval > 0 and any_interval > 0
        self.all_interval = all_interval
        self.any_interval = any_interval

    def decide(self, step: int, group_has_prefill: list[bool]) -> str:
        """→ "prefill" | "decode"（全组一致的决策；group 含本 rank）。"""
        assert group_has_prefill, "至少一个 rank"
        any_work = any(group_has_prefill)
        all_work = all(group_has_prefill)
        if not any_work:
            return "decode"  # 空手兜底（防死循环）
        if step % self.any_interval == 0:
            return "prefill"  # ANY 点：任一有活 → 集体 prefill（防饿死）
        if step % self.all_interval == 0:
            return "prefill" if all_work else "decode"  # ALL 点：全员一致
        return "prefill" if any_work else "decode"

    @staticmethod
    def exchange_flags(local_has_prefill: bool, group=None) -> list[bool]:
        """两轮 gloo allreduce 的本地等价：交换各 rank 的标志。
        集群用法：dist.all_gather_object；无分布式（单 rank）→ [local]。

        审查 #15：分布式运行时错误**不再吞掉**——静默降级为 [local] 会让
        本 rank 与其他 rank 的 step 型决策分歧 → DP 组集体通信 hang。
        只有「torch.distributed 不可用/未初始化」才是合法的单 rank 回退。
        """
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - 无 torch 的 CPU 环境
            return [local_has_prefill]
        if dist.is_available() and dist.is_initialized():
            flags = [None] * dist.get_world_size(group=group)
            dist.all_gather_object(flags, local_has_prefill, group=group)
            return [bool(f) for f in flags]
        return [local_has_prefill]


__all__ = ["DPStepVoter"]

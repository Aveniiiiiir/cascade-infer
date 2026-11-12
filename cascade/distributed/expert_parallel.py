"""EP（专家并行）分发/合并：NCCL all_to_all_single（DeepEP 的通用替代）。

设计：**meta 与 hidden 拼接同行传输**（每行 [H + 3]：源 token 序 +
本地专家 id + 权重）——dispatch/combine 各一次 all_to_all，通道对称，
无需独立 meta 通道。

- dispatch：token×topk 展平 → 按专家所属 rank 稳定分桶 → all_to_all
  （等尺寸分块 padding）→ 远端得到 (行, meta)
- combine：本地专家输出拼接 meta → 反向 all_to_all → 按源 token 序
  **index_add 单次归约**（批不变第 4 要素：DeepInfer userbuffer.py:79
  语义——单次归约，不做拆分多次归约）
- 共享专家输出由调用方（mlp 路径）直接相加（combine bias 语义）

需要 torch.distributed + NCCL（集群）；Windows 导入安全（惰性）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from cascade.common.logger import get_logger

_log = get_logger("distributed.expert_parallel")

_META_DIM = 3  # 源 token 序 / 本地专家 id / 权重


class ExpertParallelGroup:
    """单 EP 组（rank i 持有 physical 专家 [i*local, (i+1)*local)）。"""

    __slots__ = ("rank", "world_size", "group", "num_physical", "local_experts")

    def __init__(self, rank: int, world_size: int, group=None, num_physical: int = 64) -> None:
        import torch.distributed as dist

        assert dist.is_available() and dist.is_initialized()
        assert num_physical % world_size == 0
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.num_physical = num_physical
        self.local_experts = num_physical // world_size

    # ------------------------------------------------------------------ dispatch

    def dispatch(
        self,
        hidden: "torch.Tensor",  # [T, H]（本 rank 的 token）
        topk_physical_ids: "torch.Tensor",  # [T, K] int64（physical 专家 id）
        topk_weight: "torch.Tensor",  # [T, K] fp32
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """→ (recv_rows [T_r, H+3], recv_counts [W])。

        recv_rows 的 meta 列 = [源 token 序, 本地专家 id, 权重]。
        """
        import torch
        from torch.distributed import all_to_all_single

        T, H = hidden.shape
        W = self.world_size
        K = topk_physical_ids.shape[1]
        flat_e = topk_physical_ids.reshape(-1)
        dst = flat_e // self.local_experts
        order = torch.argsort(dst, stable=True)
        token_idx = (torch.arange(T * K, device=hidden.device) // K)[order]  # 展平槽 → 源 token 序
        # 行拼接 [T*K, H+3]
        meta = torch.stack(
            [
                token_idx.to(torch.float32),
                (flat_e[order] % self.local_experts).to(torch.float32),
                topk_weight.reshape(-1)[order],
            ],
            dim=-1,
        ).to(hidden.dtype)
        rows = torch.cat((hidden[token_idx.long()], meta), dim=-1).contiguous()
        send_counts = torch.bincount(dst, minlength=W).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        all_to_all_single(recv_counts, send_counts, group=self.group)
        # 等尺寸分块 all_to_all
        max_send = int(send_counts.max().item())
        max_recv = int(recv_counts.max().item())
        pad = max(max_send, max_recv)
        padded = torch.zeros(W, pad, H + _META_DIM, dtype=hidden.dtype, device=hidden.device)
        offset = 0
        for r in range(W):
            c = int(send_counts[r].item())
            if c > 0:
                padded[r, :c] = rows[offset : offset + c]
                offset += c
        recv_padded = torch.zeros_like(padded)
        all_to_all_single(recv_padded.view(W, -1), padded.view(W, -1), group=self.group)
        chunks = [recv_padded[r, : int(recv_counts[r].item())] for r in range(W)]
        recv_rows = torch.cat([c for c in chunks if c.shape[0] > 0], dim=0) if any(
            c.shape[0] > 0 for c in chunks
        ) else rows[:0]
        return recv_rows, recv_counts

    # ------------------------------------------------------------------ combine

    def combine(
        self,
        expert_out: "torch.Tensor",  # [T_r, H+3]：本地专家输出 + dispatch 的 meta 原样回传
        recv_counts: "torch.Tensor",  # dispatch 返回（分块尺寸，对称回传）
        orig_T: int,
        H: int,
    ) -> "torch.Tensor":
        """反向 all_to_all + 单次归约 → [orig_T, H]。"""
        import torch
        from torch.distributed import all_to_all_single

        W = self.world_size
        T_r = expert_out.shape[0]
        pad = int(recv_counts.max().item()) if T_r > 0 else 0
        padded = torch.zeros(W, pad, H + _META_DIM, dtype=expert_out.dtype, device=expert_out.device)
        offset = 0
        for r in range(W):
            c = int(recv_counts[r].item())
            if c > 0:
                padded[r, :c] = expert_out[offset : offset + c]
                offset += c
        recv_padded = torch.zeros_like(padded)
        all_to_all_single(recv_padded.view(W, -1), padded.view(W, -1), group=self.group)
        # 拼回本 rank 的行 + 单次 index_add 归约（批不变：一次完成 topk 聚合）
        own = []
        for r in range(W):
            c = int(recv_counts[r].item())  # 分块对称（对方按我们的 recv_counts 发送）
            if c > 0:
                own.append(recv_padded[r, :c])
        rows = torch.cat([c for c in own if c.shape[0] > 0], dim=0) if any(
            c.shape[0] > 0 for c in own
        ) else expert_out[:0]
        if rows.shape[0] == 0:
            return torch.zeros(orig_T, H, dtype=expert_out.dtype, device=expert_out.device)
        token_idx = rows[:, H].long()
        weights = rows[:, H + 2].unsqueeze(-1)
        out = torch.zeros(orig_T, H, dtype=expert_out.dtype, device=expert_out.device)
        out.index_add_(0, token_idx, rows[:, :H] * weights)
        return out


__all__ = ["ExpertParallelGroup", "_META_DIM"]

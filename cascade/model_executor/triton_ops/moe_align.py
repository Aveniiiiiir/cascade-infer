"""moe_align_block_size：topk 路由 → 分桶排序（fused grouped GEMM 的输入布局）。

对应 vLLM moe_align_block_size 语义（DeepInfer 的分组 GEMM 用 DeepGEMM
的 contiguous psum 布局，同构）：

- ``sorted_token_ids``：按专家分桶的 token 槽位（-1 填充到 block 对齐），
  槶顺序 = 专家升序，桶内 token 序稳定（确定性！）；
- ``expert_ids``：每个 M-block 所属专家；
- ``num_tokens_post_pad``：填充后的总槽数。

实现为**确定性 torch 组合**（bincount + stable argsort + 向量化 scatter）：
桶内顺序与运行无关 —— 批不变的前提（vLLM 的原子版顺序不确定，会引入
浮点归约顺序漂移）。性能版 Triton kernel（无原子两阶段）在性能剖析
证明必要时替换；当前 token 规模（≤ 每步 max_batched_tokens×topk）
下 torch 路径足够。
"""

from __future__ import annotations

import torch


def moe_align_block_size(
    topk_ids: torch.Tensor,  # [T, K] int32/int64：每 token 的 top-k 专家
    num_experts: int,
    block_size: int,
    pad_to: int | None = None,  # 输出固定尺寸（CUDA graph 捕获用；None=精确尺寸）
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pad_to 给定时为**图捕获安全**路径：sorted/expert_ids 尺寸固定、
    无 .item() 同步；超出实际填充的块经 searchsorted 得到安全范围内的
    专家索引（其 sorted 槽位为 -1，kernel 侧掩码跳过）。"""
    assert topk_ids.ndim == 2
    assert block_size > 0 and num_experts > 0
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)  # [E]
    padded = (counts + block_size - 1) // block_size * block_size
    seg_end = padded.cumsum(0)  # 每专家段结束（含填充）
    seg_start = seg_end - padded
    total_padded_t = seg_end[-1]  # 标量张量（图模式下不 .item()）
    if pad_to is not None:
        out_size = pad_to
        num_blocks = (pad_to + block_size - 1) // block_size
    else:
        total_padded = int(total_padded_t.item())
        out_size = total_padded
        num_blocks = (total_padded + block_size - 1) // block_size

    # 稳定排序：专家升序、桶内 flat 序
    order = torch.argsort(flat, stable=True)  # [T*K]
    counts_prefix = counts.cumsum(0) - counts  # 每专家的 token 起始序号
    rank_in_expert = torch.arange(flat.numel(), device=device) - counts_prefix[flat]
    dest = seg_start[flat] + rank_in_expert

    sorted_token_ids = torch.full((out_size,), -1, dtype=torch.int32, device=device)
    sorted_token_ids[dest] = torch.arange(flat.numel(), dtype=torch.int32, device=device)

    if num_blocks > 0:
        block_starts = torch.arange(num_blocks, device=device) * block_size
        expert_ids = (
            torch.searchsorted(seg_start, block_starts, right=True) - 1
        ).to(torch.int32)
    else:
        expert_ids = torch.empty(0, dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, total_padded_t.to(torch.int32)


def moe_align_ref(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """朴素参考实现（纯 Python 循环；测试 oracle）。"""
    T, K = topk_ids.shape
    flat = topk_ids.reshape(-1).tolist()
    E, bs = num_experts, block_size
    buckets: list[list[int]] = [[] for _ in range(E)]
    for slot, e in enumerate(flat):
        buckets[e].append(slot)
    seg_start: list[int] = []
    acc = 0
    counts = [len(b) for b in buckets]
    padded_counts = [(c + bs - 1) // bs * bs for c in counts]
    for pc in padded_counts:
        seg_start.append(acc)
        acc += pc
    total = acc
    out_size = pad_to if pad_to is not None else total
    sorted_ids = [-1] * out_size
    for e in range(E):
        for rank, slot in enumerate(buckets[e]):
            sorted_ids[seg_start[e] + rank] = slot
    num_blocks = (total + bs - 1) // bs
    expert_ids = []
    for b in range(num_blocks):
        s = b * bs
        e_found = -1
        for e in range(E):
            if padded_counts[e] > 0 and seg_start[e] <= s < seg_start[e] + padded_counts[e]:
                e_found = e
                break
        expert_ids.append(e_found)
    return (
        torch.tensor(sorted_ids, dtype=torch.int32),
        torch.tensor(expert_ids, dtype=torch.int32),
        torch.tensor(total, dtype=torch.int32),
    )


__all__ = ["moe_align_block_size", "moe_align_ref"]

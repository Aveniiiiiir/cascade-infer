"""fused_moe：分组 GEMM 双 kernel（vLLM fused_moe 结构，确定性变体）。

流程（DeepseekMoE.forward 的 fused 路径）：
1. moe_align_block_size(topk_ids, E, BLOCK_M) → sorted_token_ids（flat 槽位
   t*K+k，-1 填充）/ expert_ids（每 M 块的专家）
2. gemm1 ×2（gate 与 up 各一次，共享 sorted 布局；免权重拼接、零额外显存）：
   inter1[s, n] = x[t] @ w{1,3}[e].T   （t = s//K，e = 该块专家）
3. swiglu：inter2 = silu(inter1[:, :N]) * inter1[:, N:]
4. gemm2：c2[s, h] = (inter2[s] @ w2[e].T) * route_weight[s]
5. 确定性归约：out[t] = Σ_k c2[t*K+k]（固定 k 升序求和，无原子）

批不变设计：全程无原子（排序由 moe_align 的 stable-argsort 保证确定性），
浮点加法顺序固定（k 升序）⇒ 相同输入 bit 级可复现。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cascade.model_executor.triton_ops.moe_align import moe_align_block_size
from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_DEFAULT_BLOCK_M = 16
_DEFAULT_BLOCK_N = 64
_DEFAULT_BLOCK_K = 64


if triton is not None:

    @triton.jit
    def _moe_gemm1_kernel(
        x_ptr,  # [T, H] 激活
        w_ptr,  # [E, N, H] 专家权重（gate 或 up）
        inter1_ptr,  # [T*K, 2N] 输出（gate 写 [:N]，up 写 [N:]）
        sorted_token_ids_ptr,  # [P] int32 flat 槽位（-1 填充）
        expert_ids_ptr,  # [num_blocks] int32
        H,
        N,
        K,
        OUT_COL_OFFSET,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tokens = tl.load(sorted_token_ids_ptr + offs_m)  # [BM] flat 槽位
        valid = tokens >= 0
        token_rows = tl.where(valid, tokens // K, 0)
        expert = tl.load(expert_ids_ptr + pid_m)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, H, BLOCK_K):
            kk = k0 + offs_k
            a = tl.load(
                x_ptr + token_rows[:, None] * H + kk[None, :],
                mask=valid[:, None] & (kk[None, :] < H),
                other=0.0,
            )  # [BM, BK]
            b = tl.load(
                w_ptr + expert.to(tl.int64) * N * H + offs_n[:, None] * H + kk[None, :],
                mask=(offs_n[:, None] < N) & (kk[None, :] < H),
                other=0.0,
            )  # [BN, BK]
            acc += tl.dot(a, tl.trans(b))
        c = acc.to(inter1_ptr.dtype.element_ty)
        out_cols = OUT_COL_OFFSET + offs_n
        tl.store(
            inter1_ptr + tokens[:, None].to(tl.int64) * (2 * N) + out_cols[None, :],
            c,
            mask=valid[:, None] & (offs_n[None, :] < N),
        )

    @triton.jit
    def _moe_gemm2_kernel(
        inter2_ptr,  # [T*K, N]（已 swiglu）
        w2_ptr,  # [E, H, N]
        c2_ptr,  # [T*K, H]
        weights_ptr,  # [T*K] fp32 路由权重（已含 scaling/norm）
        sorted_token_ids_ptr,
        expert_ids_ptr,
        H,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tokens = tl.load(sorted_token_ids_ptr + offs_m)
        valid = tokens >= 0
        expert = tl.load(expert_ids_ptr + pid_m)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # H 空间
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, N, BLOCK_K):
            kk = k0 + offs_k
            a = tl.load(
                inter2_ptr + tokens[:, None].to(tl.int64) * N + kk[None, :],
                mask=valid[:, None] & (kk[None, :] < N),
                other=0.0,
            )
            b = tl.load(
                w2_ptr + expert.to(tl.int64) * H * N + offs_n[:, None] * N + kk[None, :],
                mask=(offs_n[:, None] < H) & (kk[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, tl.trans(b))
        w = tl.load(weights_ptr + tokens, mask=valid, other=0.0)  # [BM] fp32
        c = (acc * w[:, None]).to(c2_ptr.dtype.element_ty)
        tl.store(
            c2_ptr + tokens[:, None].to(tl.int64) * H + offs_n[None, :],
            c,
            mask=valid[:, None] & (offs_n[None, :] < H),
        )


def fused_experts(
    x: torch.Tensor,  # [T, H]
    w1: torch.Tensor,  # [E, N, H]（gate）
    w3: torch.Tensor,  # [E, N, H]（up）
    w2: torch.Tensor,  # [E, H, N]（down）
    topk_weight: torch.Tensor,  # [T, K] fp32（已含 routed_scaling/norm）
    topk_ids: torch.Tensor,  # [T, K] int32/int64
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    pad_to: int | None = None,  # 图捕获模式：sorted 布局固定尺寸（≥ 最坏填充）
) -> torch.Tensor:
    """→ [T, H]（不含共享专家；调用方自行相加）。

    pad_to 给定时为图捕获安全路径：moe_align 尺寸固定、grid 固定
    （num_blocks 由 expert_ids 形状决定，无 .item() 同步）。
    """
    assert x.ndim == 2 and w1.ndim == 3 and w2.ndim == 3
    T, H = x.shape
    E, N, H2 = w1.shape
    assert H2 == H and w3.shape == w1.shape and w2.shape == (E, H, N)
    assert x.dtype == w1.dtype == w2.dtype
    if T == 0:
        return torch.zeros_like(x)
    if not (use_triton(x.device) and x.dtype in (torch.bfloat16, torch.float16)):
        return fused_experts_ref(x, w1, w3, w2, topk_weight, topk_ids)
    K = topk_ids.shape[1]
    device = x.device
    sorted_ids, expert_ids, _n_post = moe_align_block_size(topk_ids, E, block_m, pad_to=pad_to)
    num_blocks = expert_ids.numel()  # 形状决定（图模式下固定）
    inter1 = torch.empty(T * K, 2 * N, dtype=x.dtype, device=device)
    inter2 = torch.empty(T * K, N, dtype=x.dtype, device=device)
    c2 = torch.empty(T * K, H, dtype=x.dtype, device=device)
    sorted_ids = sorted_ids.to(torch.int32)
    expert_ids_c = expert_ids.to(torch.int32)
    weights_flat = topk_weight.reshape(-1).to(torch.float32).contiguous()
    grid1 = (num_blocks, (N + block_n - 1) // block_n)
    for w_ptr, col_offset in ((w1, 0), (w3, N)):
        _moe_gemm1_kernel[grid1](
            x, w_ptr, inter1, sorted_ids, expert_ids_c,
            H, N, K, col_offset,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
        )
    inter2.copy_(F.silu(inter1[:, :N]) * inter1[:, N:])
    grid2 = (num_blocks, (H + block_n - 1) // block_n)
    _moe_gemm2_kernel[grid2](
        inter2, w2, c2, weights_flat, sorted_ids, expert_ids_c,
        H, N, K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    # 确定性归约：固定 k 升序求和
    out = c2.view(T, K, H).sum(dim=1)
    return out.to(x.dtype)


def fused_experts_ref(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """参考实现（逐专家循环；堆叠权重）。"""
    T, H = x.shape
    E, N, _ = w1.shape
    K = topk_ids.shape[1]
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    for e in range(E):
        token_rows, k_slots = torch.nonzero(topk_ids == e, as_tuple=True)
        if token_rows.numel() == 0:
            continue
        rows = torch.unique(token_rows)
        h1 = x[rows] @ w1[e].T
        h3 = x[rows] @ w3[e].T
        act = F.silu(h1) * h3
        expert_out = act @ w2[e].T  # [R, H]
        weights = torch.zeros(T, device=x.device, dtype=torch.float32)
        weights[token_rows] = topk_weight[token_rows, k_slots]
        out.index_add_(
            0, rows, (expert_out * weights[rows].to(x.dtype).unsqueeze(-1)).to(x.dtype)
        )
    return out


__all__ = ["fused_experts", "fused_experts_ref"]

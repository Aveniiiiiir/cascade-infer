"""paged_mla_decode_attn：吸收式 decode 注意力（split-K / FlashDecoding）。

KV 只需 latent（c‖rope，576 维）；q 为已吸收形态（q_abs‖q_rope，576 维）。
输出为**加权 latent** [N, H, kv_lora]，W_UV 上投影由调用方（mla.py）完成
（vLLM triton MLA decode 同构）。

结构：两 kernel ——
1. partial：grid (N, num_splits, H/BLOCK_H)，每程序处理一个 seq 的一个
   KV 分片，在线 softmax 累积 [BLOCK_H, kv_lora]，输出归一化 partial v
   与 lse；
2. reduce：grid (N, H/BLOCK_H)，按 lse softmax 权重合并分片。

数值契约：dot 用 bf16 输入 fp32 累积（概率权重 p 转 bf16 参与 dot ——
vLLM 生产路径同款），kernel vs ref 用 bf16 级容差（rtol 2e-2）。
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_MAX_SPLITS = 8
_DEFAULT_BLOCK_H = 16
_DEFAULT_BLOCK_N = 32


if triton is not None:

    @triton.jit
    def _decode_partial_kernel(
        q_ptr,  # [N, H, D] bf16（D = kv_lora + rope）
        kv_ptr,  # [num_blocks, BS, D] bf16
        block_tables_ptr,  # [N, MAX_BLOCKS] int32
        seqlens_ptr,  # [N] int64
        partial_v_ptr,  # [N, SPLITS, H, KV_LORA] fp32
        partial_lse_ptr,  # [N, SPLITS, H] fp32
        H,
        MAX_BLOCKS,
        BLOCKS_PER_SPLIT,
        SCALE,
        BS: tl.constexpr,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        n = tl.program_id(0)
        sp = tl.program_id(1)
        hb = tl.program_id(2)
        D: tl.constexpr = KV_LORA + ROPE
        seqlen = tl.load(seqlens_ptr + n)
        start = sp * BLOCKS_PER_SPLIT * BS
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, KV_LORA)
        offs_r = tl.arange(0, ROPE)
        if start >= seqlen:
            # 空分片：lse = -inf（reduce 时权重为 0），v = 0
            lse = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            tl.store(partial_lse_ptr + (n * tl.num_programs(1) + sp) * H + offs_h, lse)
            v = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
            tl.store(
                partial_v_ptr + ((n * tl.num_programs(1) + sp) * H + offs_h[:, None]) * KV_LORA + offs_d[None, :],
                v,
            )
            return
        end = tl.minimum(start + BLOCKS_PER_SPLIT * BS, seqlen)
        # q tiles（每程序一次加载）
        q_base = q_ptr + n * H * D + offs_h[:, None] * D
        q_abs = tl.load(q_base + offs_d[None, :])  # [BLOCK_H, KV_LORA] bf16
        q_rope = tl.load(q_base + KV_LORA + offs_r[None, :])  # [BLOCK_H, ROPE]
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
        # KV 分片内按 BLOCK_N 展平迭代（BLOCK_N 整除 BS ⇒ 每 chunk 恰在一个块内）
        for t0 in range(start, end, BLOCK_N):
            block_idx = t0 // BS
            block_id = tl.load(block_tables_ptr + n * MAX_BLOCKS + block_idx)
            base = block_id * BS * D
            offs_n = (t0 % BS) + tl.arange(0, BLOCK_N)
            tok = t0 + tl.arange(0, BLOCK_N)
            valid = tok < end
            c = tl.load(
                kv_ptr + base + offs_n[:, None] * D + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BLOCK_N, KV_LORA] bf16
            rope = tl.load(
                kv_ptr + base + offs_n[:, None] * D + KV_LORA + offs_r[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BLOCK_N, ROPE]
            s = tl.dot(q_abs, tl.trans(c)) + tl.dot(q_rope, tl.trans(rope))  # [H, N] fp32
            s = s * SCALE
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(c.dtype), c)
            m_i = m_new
        v = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)
        split_idx = n * tl.num_programs(1) + sp
        tl.store(partial_lse_ptr + split_idx * H + offs_h, lse)
        tl.store(
            partial_v_ptr + (split_idx * H + offs_h[:, None]) * KV_LORA + offs_d[None, :], v
        )

    @triton.jit
    def _decode_reduce_kernel(
        partial_v_ptr,  # [N, SPLITS, H, KV_LORA] fp32
        partial_lse_ptr,  # [N, SPLITS, H] fp32
        out_ptr,  # [N, H, KV_LORA] fp32
        H,
        SPLITS,
        KV_LORA: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        n = tl.program_id(0)
        hb = tl.program_id(1)
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, KV_LORA)
        m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        for sp in range(SPLITS):
            m = tl.maximum(m, tl.load(partial_lse_ptr + (n * SPLITS + sp) * H + offs_h))
        w_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        out = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
        for sp in range(SPLITS):
            lse = tl.load(partial_lse_ptr + (n * SPLITS + sp) * H + offs_h)
            w = tl.exp(lse - m)
            v = tl.load(
                partial_v_ptr + ((n * SPLITS + sp) * H + offs_h[:, None]) * KV_LORA + offs_d[None, :]
            )
            out += w[:, None] * v
            w_sum += w
        out = out / w_sum[:, None]
        tl.store(out_ptr + (n * H + offs_h[:, None]) * KV_LORA + offs_d[None, :], out)


def paged_mla_decode_attn(
    q: torch.Tensor,  # [N, H, D] bf16（已吸收：q_abs‖q_rope）
    kv: torch.Tensor,  # [num_blocks, BS, D] bf16 本层视图
    block_tables: torch.Tensor,  # [N, max_blocks] int32/int64
    seqlens: torch.Tensor,  # [N] int64
    scale: float,
    kv_lora: int,
    num_splits: int | None = None,
    block_h: int = _DEFAULT_BLOCK_H,
    block_n: int = _DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """→ 加权 latent [N, H, kv_lora] fp32。"""
    assert q.ndim == 3 and kv.ndim == 3
    N, H, D = q.shape
    BS = kv.shape[1]
    assert kv.shape[2] == D
    assert D - kv_lora > 0 and D - kv_lora <= 64
    assert BS % block_n == 0, f"block_n={block_n} 必须整除 kv block size {BS}"
    assert q.dtype == kv.dtype and q.is_contiguous() and kv.is_contiguous()
    # kernel 路径要求 tensor-core dtype；fp32（紧容差测试场景）走 ref
    if not (use_triton(q.device) and q.dtype in (torch.bfloat16, torch.float16)):
        return paged_mla_decode_attn_ref(q, kv, block_tables, seqlens, scale, kv_lora)
    max_seqlen = int(seqlens.max().item())
    blocks_needed = (max_seqlen + BS - 1) // BS
    if num_splits is None:
        num_splits = max(1, min(_MAX_SPLITS, blocks_needed // 4))
    blocks_per_split = (blocks_needed + num_splits - 1) // num_splits
    bt = block_tables.to(torch.int32)
    partial_v = torch.empty(N, num_splits, H, kv_lora, dtype=torch.float32, device=q.device)
    partial_lse = torch.empty(N, num_splits, H, dtype=torch.float32, device=q.device)
    out = torch.empty(N, H, kv_lora, dtype=torch.float32, device=q.device)
    grid = (N, num_splits, (H + block_h - 1) // block_h)
    _decode_partial_kernel[grid](
        q, kv, bt, seqlens, partial_v, partial_lse,
        H, bt.shape[1], blocks_per_split, scale,
        BS=BS, KV_LORA=kv_lora, ROPE=D - kv_lora,
        BLOCK_H=block_h, BLOCK_N=block_n, num_warps=8,
    )
    grid2 = (N, (H + block_h - 1) // block_h)
    _decode_reduce_kernel[grid2](
        partial_v, partial_lse, out, H, num_splits, KV_LORA=kv_lora, BLOCK_H=block_h
    )
    return out


def paged_mla_decode_attn_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    scale: float,
    kv_lora: int,
) -> torch.Tensor:
    """PyTorch 参考实现（逐 seq；kernel 的容差级 oracle）。"""
    N, H, D = q.shape
    BS = kv.shape[1]
    out = torch.empty(N, H, kv_lora, dtype=torch.float32, device=q.device)
    for n in range(N):
        P = int(seqlens[n])
        nblocks = (P + BS - 1) // BS
        rows = kv[block_tables[n, :nblocks].long()].reshape(-1, D)[:P]  # [P, D]
        s = torch.einsum("hd,pd->hp", q[n].float(), rows.float()) * scale  # [H, P]
        w = torch.softmax(s, dim=-1)
        out[n] = torch.einsum("hp,pk->hk", w, rows[:, :kv_lora].float())
    return out


__all__ = ["paged_mla_decode_attn", "paged_mla_decode_attn_ref"]

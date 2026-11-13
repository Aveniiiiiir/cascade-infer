"""paged_mla_prefill_attn：吸收式变长因果 prefill 注意力。

段 = 一个 scheduled seq 的本步 token（chunked prefill 的中段天然正确：
q 位置为绝对位置，因果掩码 pos_q >= pos_k，KV 经 block 表取 [0, kv_end)）。

grid (num_q_blocks, H)：每程序处理 BLOCK_M 个 q token × 1 个 head。
输出加权 latent [T, H, kv_lora]（W_UV 上投影由调用方完成）。

q 块元数据（wrapper 构建）：
- qo_seg:  块所属 seq 行（block_tables 行号）
- qo_qbase: 块首 token 在展平 q 中的行号
- qo_pos:  块首 token 的绝对位置
- qo_len:  块内 q token 数（尾块可能 < BLOCK_M）
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_DEFAULT_BLOCK_M = 16
_DEFAULT_BLOCK_N = 32


if triton is not None:

    @triton.jit
    def _prefill_attn_kernel(
        q_ptr,  # [T, H, D] bf16
        kv_ptr,  # [num_blocks, BS, D] bf16
        block_tables_ptr,  # [num_seqs, MAX_BLOCKS] int32
        qo_seg_ptr,  # [num_q_blocks] int32
        qo_qbase_ptr,  # [num_q_blocks] int32
        qo_pos_ptr,  # [num_q_blocks] int64（绝对位置）
        qo_len_ptr,  # [num_q_blocks] int32
        seqlens_ptr,  # [num_seqs] int64
        out_ptr,  # [T, H, KV_LORA] fp32
        H,
        MAX_BLOCKS,
        SCALE,
        BS: tl.constexpr,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        ROPE_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        D: tl.constexpr = KV_LORA + ROPE
        seg = tl.load(qo_seg_ptr + pid_m)
        qbase = tl.load(qo_qbase_ptr + pid_m)
        q_pos0 = tl.load(qo_pos_ptr + pid_m)
        q_len = tl.load(qo_len_ptr + pid_m)
        seqlen = tl.load(seqlens_ptr + seg)
        kv_end = tl.minimum(q_pos0 + q_len, seqlen)  # 因果上界（含本块 q 的 kv）
        offs_m = tl.arange(0, BLOCK_M)
        m_valid = offs_m < q_len
        offs_d = tl.arange(0, KV_LORA)
        # ROPE pad（tl.dot 的 K 维下限 16）：ROPE_PAD = max(16, next_pow2(ROPE))，
        # 生产模型 ROPE=64 时取 64（不截断），tiny 模型 ROPE=8 时 pad 到 16
        offs_r = tl.arange(0, ROPE_PAD)
        rope_valid = offs_r < ROPE
        q_row = qbase + offs_m
        q_base_p = q_ptr + q_row[:, None].to(tl.int64) * (H * D) + h * D
        q_abs = tl.load(q_base_p + offs_d[None, :], mask=m_valid[:, None], other=0.0)
        q_rope = tl.load(q_base_p + KV_LORA + offs_r[None, :], mask=m_valid[:, None] & rope_valid[None, :], other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, KV_LORA], dtype=tl.float32)
        pos_q = q_pos0 + offs_m  # [BLOCK_M] 绝对位置
        # 块 id 逐 token 查表（同 decode kernel）：BLOCK_N 可跨物理块
        for t0 in range(0, kv_end, BLOCK_N):
            tok = t0 + tl.arange(0, BLOCK_N)
            valid = tok < kv_end
            blk = tl.load(
                block_tables_ptr + seg * MAX_BLOCKS + tok // BS, mask=valid, other=0
            ).to(tl.int64)
            row_base = blk * BS * D + (tok % BS) * D
            c = tl.load(
                kv_ptr + row_base[:, None] + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            rope = tl.load(
                kv_ptr + row_base[:, None] + KV_LORA + offs_r[None, :],
                mask=valid[:, None] & rope_valid[None, :],
                other=0.0,
            )
            s = tl.dot(q_abs, tl.trans(c)) + tl.dot(q_rope, tl.trans(rope))  # [M, N]
            s = s * SCALE
            causal = (pos_q[:, None] >= tok[None, :]) & m_valid[:, None] & valid[None, :]
            s = tl.where(causal, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(c.dtype), c)
            m_i = m_new
        v = acc / l_i[:, None]
        out_base = out_ptr + q_row[:, None].to(tl.int64) * (H * KV_LORA) + h * KV_LORA
        tl.store(out_base + offs_d[None, :], v, mask=m_valid[:, None])


def paged_mla_prefill_attn(
    q: torch.Tensor,  # [T, H, D] bf16（已吸收 q_abs‖q_rope，已 rope）
    kv: torch.Tensor,  # [num_blocks, BS, D] bf16
    block_tables: torch.Tensor,  # [num_seqs, max_blocks] int32/int64
    seqlens: torch.Tensor,  # [num_seqs] int64：各 seq 的 KV 总数（含历史 chunk）
    segments: list[tuple[int, int, int]],  # (seq 行, 段首绝对位置, 段长)
    scale: float,
    kv_lora: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """→ 加权 latent [T, H, kv_lora] fp32（行序与 q 一致）。"""
    assert q.ndim == 3 and kv.ndim == 3
    T, H, D = q.shape
    BS = kv.shape[1]
    assert kv.shape[2] == D and D - kv_lora > 0
    # BLOCK_N 下限 16（tl.dot 的 N 维约束）；块 id 逐 token 查表后不再要求
    # BLOCK_N 整除 KV block size
    block_n = max(16, block_n)
    rope_pad = max(16, triton.next_power_of_2(D - kv_lora))
    assert q.dtype == kv.dtype and q.is_contiguous() and kv.is_contiguous()
    # kernel 路径要求 tensor-core dtype；fp32（紧容差测试场景）走 ref
    if not (use_triton(q.device) and q.dtype in (torch.bfloat16, torch.float16)):
        return paged_mla_prefill_attn_ref(q, kv, block_tables, seqlens, segments, scale, kv_lora)
    device = q.device
    seg_ids: list[int] = []
    qbases: list[int] = []
    poss: list[int] = []
    lens: list[int] = []
    base = 0
    for row, pos_start, seg_len in segments:
        for j in range(0, seg_len, block_m):
            m = min(block_m, seg_len - j)
            seg_ids.append(row)
            qbases.append(base + j)
            poss.append(pos_start + j)
            lens.append(m)
        base += seg_len
    assert base == T
    out = torch.empty(T, H, kv_lora, dtype=torch.float32, device=device)
    grid = (len(qbases), H)
    _prefill_attn_kernel[grid](
        q,
        kv,
        block_tables.to(torch.int32),
        torch.tensor(seg_ids, dtype=torch.int32, device=device),
        torch.tensor(qbases, dtype=torch.int32, device=device),
        torch.tensor(poss, dtype=torch.int64, device=device),
        torch.tensor(lens, dtype=torch.int32, device=device),
        seqlens,
        out,
        H,
        block_tables.shape[1],
        scale,
        BS=BS,
        KV_LORA=kv_lora,
        ROPE=D - kv_lora,
        ROPE_PAD=rope_pad,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def paged_mla_prefill_attn_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    segments: list[tuple[int, int, int]],
    scale: float,
    kv_lora: int,
) -> torch.Tensor:
    """PyTorch 参考实现（逐段 + 因果掩码）。"""
    T, H, D = q.shape
    BS = kv.shape[1]
    out = torch.empty(T, H, kv_lora, dtype=torch.float32, device=q.device)
    base = 0
    for row, pos_start, seg_len in segments:
        P = min(pos_start + seg_len, int(seqlens[row]))
        nblocks = (P + BS - 1) // BS
        rows = kv[block_tables[row, :nblocks].long()].reshape(-1, D)[:P]  # [P, D]
        s = torch.einsum("mhd,pd->hmp", q[base : base + seg_len].float(), rows.float()) * scale
        pos_q = torch.arange(pos_start, pos_start + seg_len, device=q.device)
        pos_k = torch.arange(P, device=q.device)
        mask = pos_q[:, None] >= pos_k[None, :]
        s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(s, dim=-1)  # [H, M, P]
        out[base : base + seg_len] = torch.einsum("hmp,pk->mhk", w, rows[:, :kv_lora].float())
        base += seg_len
    return out


__all__ = ["paged_mla_prefill_attn", "paged_mla_prefill_attn_ref"]

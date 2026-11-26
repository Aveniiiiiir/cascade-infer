"""paged_mla_decode_attn：吸收式 decode 注意力（split-K / FlashDecoding）。

KV 只需 latent（c‖rope，576 维）；q 为已吸收形态（q_abs‖q_rope，576 维）。
输出为**加权 latent** [N, H, kv_lora]，W_UV 上投影由调用方（mla.py）完成
（vLLM triton MLA decode 同构）。

两种 KV 布局：
- bf16：单张量 [num_blocks, BS, KV_LORA+ROPE]（默认）
- fp8 ：c 张量 [num_blocks, BS, KV_LORA] float8_e4m3fn + rope 张量 bf16 +
  每 token fp32 scale（DeepSeek-V3 生产同构；kernel 内 dequant 到 bf16 后
  走同一组 tl.dot）——KV 带宽减半是长上下文 decode 的主要收益。

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


def _fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


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
        NUM_SPLITS,
        SCALE,
        BS: tl.constexpr,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        ROPE_PAD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        n = tl.program_id(0)
        sp = tl.program_id(1)
        hb = tl.program_id(2)
        D: tl.constexpr = KV_LORA + ROPE
        seqlen = tl.load(seqlens_ptr + n)
        # 分片边界全在设备侧计算（审查补充：此前 wrapper 用
        # seqlens.max().item() 求 blocks_per_split——捕获区间内的 CPU-GPU
        # 同步会让 CUDA graph 捕获失败，与 rope .item() 同类缺陷）
        blocks_needed = tl.cdiv(seqlen, BS)
        blocks_per_split = tl.cdiv(blocks_needed, NUM_SPLITS)
        start = sp * blocks_per_split * BS
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, KV_LORA)
        # 头维 mask + 地址 clamp：BLOCK_H=16 > H（tiny 模型 4 头）时 offs_h
        # 越界。mask 只控制是否读写，不阻止地址计算越界——用 safe 下标
        # （越界行映射到 H-1，store 由 mask 阻止）保证地址恒在分配内。
        h_valid = offs_h < H
        offs_h = tl.where(h_valid, offs_h, H - 1)
        # ROPE pad（tl.dot 的 K 维最小 16 约束）：ROPE_PAD = max(16, next_pow2(ROPE))；
        # 生产模型 ROPE=64 时 ROPE_PAD=64（不截断），tiny 模型 ROPE=8 时 pad 到 16
        offs_r = tl.arange(0, ROPE_PAD)
        rope_valid = offs_r < ROPE
        if start >= seqlen:
            # 空分片：lse = -inf（reduce 时权重为 0），v = 0
            lse = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            tl.store(partial_lse_ptr + (n * tl.num_programs(1) + sp) * H + offs_h, lse, mask=h_valid)
            v = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
            tl.store(
                partial_v_ptr + ((n * tl.num_programs(1) + sp) * H + offs_h[:, None]) * KV_LORA + offs_d[None, :],
                v,
                mask=h_valid[:, None],
            )
            return
        end = tl.minimum(start + blocks_per_split * BS, seqlen)
        # q tiles（每程序一次加载）
        q_base = q_ptr + n * H * D + offs_h[:, None] * D
        q_abs = tl.load(q_base + offs_d[None, :], mask=h_valid[:, None], other=0.0)  # [BLOCK_H, KV_LORA] bf16
        q_rope = tl.load(q_base + KV_LORA + offs_r[None, :], mask=h_valid[:, None] & rope_valid[None, :], other=0.0)  # [BLOCK_H, 16]
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
        # KV 分片内按 BLOCK_N 展平迭代。块 id **逐 token 查表**（不再假设
        # BLOCK_N 整除 BS）：小 KV block（tiny 模型 bs=8）+ BLOCK_N≥16
        # （tl.dot 的 N 维下限）时一个 chunk 可跨多个物理块。
        for t0 in range(start, end, BLOCK_N):
            tok = t0 + tl.arange(0, BLOCK_N)
            valid = tok < end
            # int64（审查 #7）：blk*BS*D 在 int32 下 ~58k 块即溢出越界读
            blk = tl.load(
                block_tables_ptr + n * MAX_BLOCKS + tok // BS, mask=valid, other=0
            ).to(tl.int64)
            row_base = blk * BS * D + (tok % BS) * D  # [BLOCK_N] 每 token 行基址
            c = tl.load(
                kv_ptr + row_base[:, None] + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BLOCK_N, KV_LORA] bf16
            rope = tl.load(
                kv_ptr + row_base[:, None] + KV_LORA + offs_r[None, :],
                mask=valid[:, None] & rope_valid[None, :],
                other=0.0,
            )  # [BLOCK_N, ROPE_PAD]
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
        tl.store(partial_lse_ptr + split_idx * H + offs_h, lse, mask=h_valid)
        tl.store(
            partial_v_ptr + (split_idx * H + offs_h[:, None]) * KV_LORA + offs_d[None, :], v,
            mask=h_valid[:, None],
        )

    @triton.jit
    def _decode_partial_kernel_fp8(
        q_ptr,  # [N, H, D] bf16
        c_ptr,  # [num_blocks, BS, KV_LORA] float8_e4m3fn
        rope_ptr,  # [num_blocks, BS, ROPE] bf16
        sc_ptr,  # [num_blocks, BS] fp32
        block_tables_ptr,  # [N, MAX_BLOCKS] int32
        seqlens_ptr,  # [N] int64
        partial_v_ptr,  # [N, SPLITS, H, KV_LORA] fp32
        partial_lse_ptr,  # [N, SPLITS, H] fp32
        H,
        MAX_BLOCKS,
        NUM_SPLITS,
        SCALE,
        BS: tl.constexpr,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        ROPE_PAD: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        n = tl.program_id(0)
        sp = tl.program_id(1)
        hb = tl.program_id(2)
        D: tl.constexpr = KV_LORA + ROPE
        GROUP: tl.constexpr = KV_LORA // NUM_GROUPS
        seqlen = tl.load(seqlens_ptr + n)
        blocks_needed = tl.cdiv(seqlen, BS)
        blocks_per_split = tl.cdiv(blocks_needed, NUM_SPLITS)
        start = sp * blocks_per_split * BS
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, KV_LORA)
        h_valid = offs_h < H
        offs_h = tl.where(h_valid, offs_h, H - 1)
        offs_r = tl.arange(0, ROPE_PAD)
        rope_valid = offs_r < ROPE
        if start >= seqlen:
            lse = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            tl.store(partial_lse_ptr + (n * tl.num_programs(1) + sp) * H + offs_h, lse, mask=h_valid)
            v = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
            tl.store(
                partial_v_ptr + ((n * tl.num_programs(1) + sp) * H + offs_h[:, None]) * KV_LORA + offs_d[None, :],
                v,
                mask=h_valid[:, None],
            )
            return
        end = tl.minimum(start + blocks_per_split * BS, seqlen)
        q_base = q_ptr + n * H * D + offs_h[:, None] * D
        q_abs = tl.load(q_base + offs_d[None, :], mask=h_valid[:, None], other=0.0)
        q_rope = tl.load(q_base + KV_LORA + offs_r[None, :], mask=h_valid[:, None] & rope_valid[None, :], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
        for t0 in range(start, end, BLOCK_N):
            tok = t0 + tl.arange(0, BLOCK_N)
            valid = tok < end
            blk = tl.load(
                block_tables_ptr + n * MAX_BLOCKS + tok // BS, mask=valid, other=0
            ).to(tl.int64)
            flat = blk * BS + (tok % BS)  # [BLOCK_N] 行号（block 内 slot 展平）
            c_raw = tl.load(
                c_ptr + flat[:, None] * KV_LORA + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BLOCK_N, KV_LORA] fp8
            offs_g = tl.arange(0, NUM_GROUPS)
            sc = tl.load(
                sc_ptr + flat[:, None] * NUM_GROUPS + offs_g[None, :],
                mask=valid[:, None],
                other=1.0,
            )  # [BLOCK_N, NUM_GROUPS] fp32
            c3 = tl.reshape(c_raw.to(tl.float32), [BLOCK_N, NUM_GROUPS, GROUP])
            c = tl.reshape(c3 * sc[:, :, None], [BLOCK_N, KV_LORA]).to(tl.bfloat16)
            rope = tl.load(
                rope_ptr + (flat[:, None] * ROPE) + offs_r[None, :],
                mask=valid[:, None] & rope_valid[None, :],
                other=0.0,
            )  # [BLOCK_N, ROPE_PAD]
            s = tl.dot(q_abs, tl.trans(c)) + tl.dot(q_rope, tl.trans(rope))
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
        tl.store(partial_lse_ptr + split_idx * H + offs_h, lse, mask=h_valid)
        tl.store(
            partial_v_ptr + (split_idx * H + offs_h[:, None]) * KV_LORA + offs_d[None, :], v,
            mask=h_valid[:, None],
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
        h_valid = offs_h < H  # 同 partial：BLOCK_H > H 时 mask 越界头
        offs_h = tl.where(h_valid, offs_h, H - 1)  # 地址 clamp（同 partial）
        m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        for sp in range(SPLITS):
            m = tl.maximum(m, tl.load(partial_lse_ptr + (n * SPLITS + sp) * H + offs_h, mask=h_valid, other=float("-inf")))
        w_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        out = tl.zeros([BLOCK_H, KV_LORA], dtype=tl.float32)
        for sp in range(SPLITS):
            lse = tl.load(partial_lse_ptr + (n * SPLITS + sp) * H + offs_h, mask=h_valid, other=float("-inf"))
            w = tl.exp(lse - m)
            v = tl.load(
                partial_v_ptr + ((n * SPLITS + sp) * H + offs_h[:, None]) * KV_LORA + offs_d[None, :],
                mask=h_valid[:, None],
                other=0.0,
            )
            out += w[:, None] * v
            w_sum += w
        out = out / w_sum[:, None]
        tl.store(out_ptr + (n * H + offs_h[:, None]) * KV_LORA + offs_d[None, :], out, mask=h_valid[:, None])


def paged_mla_decode_attn(
    q: torch.Tensor,  # [N, H, D] bf16（已吸收：q_abs‖q_rope）
    kv: torch.Tensor,  # [num_blocks, BS, D] bf16；fp8 时为 c [.., KV_LORA] fp8
    block_tables: torch.Tensor,  # [N, max_blocks] int32/int64
    seqlens: torch.Tensor,  # [N] int64
    scale: float,
    kv_lora: int,
    num_splits: int | None = None,
    block_h: int = _DEFAULT_BLOCK_H,
    block_n: int = _DEFAULT_BLOCK_N,
    rope: torch.Tensor | None = None,  # fp8：k_rope [num_blocks, BS, ROPE] bf16
    kv_scale: torch.Tensor | None = None,  # fp8：每 token scale [num_blocks, BS] fp32
) -> torch.Tensor:
    """→ 加权 latent [N, H, kv_lora] fp32。"""
    assert q.ndim == 3 and kv.ndim == 3
    N, H, D = q.shape
    BS = kv.shape[1]
    fp8 = kv.dtype == _fp8_dtype()
    num_groups = 1
    if fp8:
        assert rope is not None and kv_scale is not None, "fp8 KV 需要 rope/kv_scale"
        assert rope.shape[:2] == (kv.shape[0], BS) and kv_scale.shape[:2] == (kv.shape[0], BS)
        num_groups = kv_scale.shape[-1]
        assert kv_lora % num_groups == 0
        assert num_groups & (num_groups - 1) == 0, "分组数必须为 2 的幂（Triton reshape）"
        assert rope.dtype in (torch.bfloat16, torch.float16) and kv_scale.dtype == torch.float32
        assert kv.shape[2] == kv_lora and rope.shape[2] == D - kv_lora
    else:
        assert kv.shape[2] == D
    assert D - kv_lora > 0 and D - kv_lora <= 64
    # BLOCK_N 下限 16（tl.dot 的 N 维约束）；块 id 逐 token 查表后不再要求
    # BLOCK_N 整除 KV block size
    block_n = max(16, block_n)
    assert kv.is_contiguous() and block_tables.is_contiguous()
    # kernel 路径要求 tensor-core dtype；fp32（紧容差测试场景）走 ref。
    # ROPE<16 不再回退 ref（ref 有 .item()，图捕获不安全）——kernel 用
    # ROPE_PAD 把 rope 维 pad 到 ≥16，tiny 模型（ROPE=8）也走 kernel。
    if not (use_triton(q.device) and q.dtype in (torch.bfloat16, torch.float16)):
        return paged_mla_decode_attn_ref(q, kv, block_tables, seqlens, scale, kv_lora, rope, kv_scale)
    assert q.is_contiguous()
    rope_dim = D - kv_lora
    rope_pad = max(16, triton.next_power_of_2(rope_dim))
    # num_splits=None 仅出现在 eager 路径（可安全同步）；图捕获路径由
    # CudaGraphRunner 固定 num_splits，此处**不做 .item()**（审查补充：
    # wrapper 的 seqlens.max().item() 是捕获区间内的非法同步）
    if num_splits is None:
        max_seqlen = int(seqlens.max().item())
        blocks_needed = (max_seqlen + BS - 1) // BS
        num_splits = max(1, min(_MAX_SPLITS, blocks_needed // 4))
    bt = block_tables.to(torch.int32)
    partial_v = torch.empty(N, num_splits, H, kv_lora, dtype=torch.float32, device=q.device)
    partial_lse = torch.empty(N, num_splits, H, dtype=torch.float32, device=q.device)
    out = torch.empty(N, H, kv_lora, dtype=torch.float32, device=q.device)
    grid = (N, num_splits, (H + block_h - 1) // block_h)
    common = dict(
        BS=BS, KV_LORA=kv_lora, ROPE=rope_dim, ROPE_PAD=rope_pad,
        BLOCK_H=block_h, BLOCK_N=block_n, num_warps=8,
    )
    if fp8:
        _decode_partial_kernel_fp8[grid](
            q, kv, rope, kv_scale, bt, seqlens, partial_v, partial_lse,
            H, bt.shape[1], num_splits, scale, NUM_GROUPS=num_groups, **common,
        )
    else:
        _decode_partial_kernel[grid](
            q, kv, bt, seqlens, partial_v, partial_lse,
            H, bt.shape[1], num_splits, scale, **common,
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
    rope: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch 参考实现（逐 seq；kernel 的容差级 oracle）。

    fp8 布局：分别 gather c/rope/scale 并 dequant 成 bf16 行（与 kernel 的
    数值路径一致：fp32 乘 scale 后转 bf16）。
    """
    N, H, D = q.shape
    BS = kv.shape[1]
    out = torch.empty(N, H, kv_lora, dtype=torch.float32, device=q.device)
    fp8 = kv.dtype == _fp8_dtype()
    for n in range(N):
        P = int(seqlens[n])
        nblocks = (P + BS - 1) // BS
        rows_idx = block_tables[n, :nblocks].long()
        if fp8:
            c = (
                kv.view(torch.int8)[rows_idx]
                .view(torch.float8_e4m3fn)
                .reshape(-1, kv_lora)[:P]
                .float()
            )
            sc = kv_scale[rows_idx].reshape(-1, kv_scale.shape[-1])[:P]  # type: ignore[index]
            sc = sc.repeat_interleave(kv_lora // sc.shape[-1], dim=-1).float()
            r = rope[rows_idx].reshape(-1, rope.shape[2])[:P].float()  # type: ignore[index]
            rows = torch.cat((c * sc, r), dim=-1).to(q.dtype)  # [P, D]
        else:
            rows = kv[rows_idx].reshape(-1, D)[:P]  # [P, D]
        s = torch.einsum("hd,pd->hp", q[n].float(), rows.float()) * scale  # [H, P]
        w = torch.softmax(s, dim=-1)
        out[n] = torch.einsum("hp,pk->hk", w, rows[:, :kv_lora].float())
    return out


__all__ = ["paged_mla_decode_attn", "paged_mla_decode_attn_ref"]
"""store_paged_kv：把本步新 token 的 latent KV 写入分页缓存。

语义 oracle 即 cache_engine.store_latent_reference 的 gather 索引版：
    block = block_tables[seq_ids[t], positions[t] // bs]
    slot  = positions[t] % bs
    kv[block, slot] = latent[t]

Triton kernel: grid (T, dim/BLOCK_D)，每 program 处理一个 token 的
一个 dim 块。BLOCK_D=64 使 576 dim 恰好 9 块（无尾块掩码路径）。
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_BLOCK_D = 64


if triton is not None:

    @triton.jit
    def _store_paged_kv_kernel(
        kv_ptr,  # [num_blocks, block_size, dim] 连续
        latent_ptr,  # [T, dim] 连续
        seq_ids_ptr,  # [T] int64
        positions_ptr,  # [T] int64
        block_tables_ptr,  # [n_seqs, max_blocks] int64 连续
        max_blocks,
        dim,
        BLOCK_SIZE: tl.constexpr,  # 分页 block 大小（token 数）
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        dblock = tl.program_id(1)
        seq = tl.load(seq_ids_ptr + t)
        pos = tl.load(positions_ptr + t)
        block = tl.load(block_tables_ptr + seq * max_blocks + pos // BLOCK_SIZE)
        slot = pos % BLOCK_SIZE
        offs_d = dblock * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs_d < dim
        val = tl.load(latent_ptr + t * dim + offs_d, mask=mask)
        # kv 线性地址 = ((block * BLOCK_SIZE) + slot) * dim + d
        dst = (block * BLOCK_SIZE + slot) * dim + offs_d
        tl.store(kv_ptr + dst, val, mask=mask)

    @triton.jit
    def _store_paged_kv_fp8_kernel(
        c_ptr,  # [num_blocks, block_size, kv_lora] float8_e4m3fn（按 i8 拷贝）
        rope_ptr,  # [num_blocks, block_size, rope] bf16
        scale_ptr,  # [num_blocks, block_size] fp32
        cq_ptr,  # [T, kv_lora] float8_e4m3fn（调用方已量化）
        qscale_ptr,  # [T, NUM_GROUPS] fp32（每 128 维一个 scale）
        latent_ptr,  # [T, kv_lora + rope] bf16（rope 源）
        seq_ids_ptr,  # [T] int64
        positions_ptr,  # [T] int64
        block_tables_ptr,  # [n_seqs, max_blocks] int64
        max_blocks,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        CPAD: tl.constexpr,
        RPAD: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        GPAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """c/rope/scale 三路原地写入（c 为已量化字节，kernel 只搬运）。"""
        t = tl.program_id(0)
        seq = tl.load(seq_ids_ptr + t)
        pos = tl.load(positions_ptr + t)
        block = tl.load(block_tables_ptr + seq * max_blocks + pos // BLOCK_SIZE)
        slot = pos % BLOCK_SIZE
        row = (block * BLOCK_SIZE + slot).to(tl.int64)
        offs_c = tl.arange(0, CPAD)
        mc = offs_c < KV_LORA
        cv = tl.load(cq_ptr + t.to(tl.int64) * KV_LORA + offs_c, mask=mc)
        tl.store(c_ptr + row * KV_LORA + offs_c, cv, mask=mc)
        offs_r = tl.arange(0, RPAD)
        mr = offs_r < ROPE
        dim = KV_LORA + ROPE
        rv = tl.load(latent_ptr + t.to(tl.int64) * dim + KV_LORA + offs_r, mask=mr)
        tl.store(rope_ptr + row * ROPE + offs_r, rv, mask=mr)
        offs_g = tl.arange(0, GPAD)
        mg = offs_g < NUM_GROUPS
        sc = tl.load(qscale_ptr + t * NUM_GROUPS + offs_g, mask=mg)
        tl.store(scale_ptr + row * NUM_GROUPS + offs_g, sc, mask=mg)


def store_paged_kv(
    kv: torch.Tensor,  # [num_blocks, block_size, dim]，本层视图
    latent: torch.Tensor,  # [T, dim] 新 token 的 KV（连续）
    seq_ids: torch.Tensor,  # [T] int64
    positions: torch.Tensor,  # [T] int64 绝对位置
    block_tables: torch.Tensor,  # [n_seqs, max_blocks] int64 连续
) -> None:
    """原地写入（in-place）。dtype 任意一致即可（Triton 路径按字节搬运）。"""
    assert kv.ndim == 3 and latent.ndim == 2
    assert kv.is_contiguous() and latent.is_contiguous() and block_tables.is_contiguous()
    assert seq_ids.dtype == torch.int64 and positions.dtype == torch.int64
    assert block_tables.dtype == torch.int64
    T, dim = latent.shape
    assert kv.shape[2] == dim
    num_blocks, block_size, _ = kv.shape
    if T == 0:
        return
    if use_triton(kv.device):
        grid = (T, triton.cdiv(dim, _BLOCK_D))
        _store_paged_kv_kernel[grid](
            kv,
            latent,
            seq_ids,
            positions,
            block_tables,
            block_tables.shape[1],
            dim,
            BLOCK_SIZE=block_size,
            BLOCK_D=_BLOCK_D,
            num_warps=1,
        )
    else:
        store_paged_kv_ref(kv, latent, seq_ids, positions, block_tables)


def store_paged_kv_ref(
    kv: torch.Tensor,
    latent: torch.Tensor,
    seq_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """PyTorch 参考实现（gather 索引；Triton kernel 的 bit 级 oracle）。"""
    bs = kv.shape[1]
    blocks = block_tables[seq_ids, torch.div(positions, bs, rounding_mode="floor")]
    slots = positions % bs
    kv[blocks, slots] = latent.to(kv.dtype)


def store_paged_kv_fp8(
    c: torch.Tensor,  # [num_blocks, bs, kv_lora] float8_e4m3fn 本层视图
    rope: torch.Tensor,  # [num_blocks, bs, rope] 模型 dtype 本层视图
    scale: torch.Tensor,  # [num_blocks, bs] fp32 本层视图
    latent: torch.Tensor,  # [T, kv_lora + rope] 本步 KV（c 未量化）
    seq_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """FP8 混合布局写入：c 每 token absmax/448 量化，rope 原精度，scale 落盘。

    量化在 PyTorch 侧完成（向量化、图捕获安全）；Triton kernel 只做三路
    字节搬运——避免在 sm89 上依赖 Triton 的 bf16→fp8 cast 指令支持
    （PTX cvt.e4m3 在 Ada 可用，但 Triton 版本差异大）。
    """
    T = latent.shape[0]
    if T == 0:
        return
    assert latent.is_contiguous() and block_tables.is_contiguous()
    kv_lora = c.shape[-1]
    rope_dim = rope.shape[-1]
    assert latent.shape[1] == kv_lora + rope_dim and rope.shape[2] == rope_dim
    assert c.dtype == torch.float8_e4m3fn and rope.dtype in (torch.bfloat16, torch.float16)
    from cascade.worker.kv_layout import FP8_MAX, fp8_group_size

    group = fp8_group_size(kv_lora)
    num_groups = kv_lora // group
    assert scale.shape[-1] == num_groups, (scale.shape, num_groups)
    cf = latent[:, :kv_lora].float().view(T, num_groups, group)
    qscale = (cf.abs().amax(dim=-1) / FP8_MAX).clamp_min(1e-12)  # [T, G]
    cq = (
        (cf / qscale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(T, kv_lora)
    )
    if use_triton(c.device):
        bs = c.shape[1]
        _store_paged_kv_fp8_kernel[(T,)](
            c.view(torch.int8),
            rope,
            scale,
            cq.view(torch.int8),
            qscale,
            latent,
            seq_ids,
            positions,
            block_tables,
            block_tables.shape[1],
            KV_LORA=kv_lora,
            ROPE=rope_dim,
            CPAD=triton.next_power_of_2(kv_lora),
            RPAD=triton.next_power_of_2(rope_dim),
            NUM_GROUPS=num_groups,
            GPAD=triton.next_power_of_2(num_groups),
            BLOCK_SIZE=bs,
            num_warps=4,
        )
    else:
        bs = c.shape[1]
        blocks = block_tables[seq_ids, torch.div(positions, bs, rounding_mode="floor")]
        slots = positions % bs
        c[blocks, slots] = cq
        rope[blocks, slots] = latent[:, kv_lora:].to(rope.dtype)
        scale[blocks, slots] = qscale


__all__ = ["store_paged_kv", "store_paged_kv_ref", "store_paged_kv_fp8"]

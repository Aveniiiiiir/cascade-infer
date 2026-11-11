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


__all__ = ["store_paged_kv", "store_paged_kv_ref"]

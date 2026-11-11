"""store_paged_kv：kernel vs ref 对拍（CUDA 上走 Triton 路径）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.store_paged_kv import (
    store_paged_kv,
    store_paged_kv_ref,
)


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_case(device: str, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    num_blocks, bs, dim = 40, 8, 576
    n_seqs, max_blocks = 4, 6
    T = 37
    kv = torch.zeros(num_blocks, bs, dim, dtype=dtype, device=device)
    latent = torch.randn(T, dim, generator=g, dtype=torch.float32).to(dtype).to(device)
    # 每 seq 独占 block 区段（(seq,pos)→(block,slot) 单射，写入确定性）
    block_tables = torch.zeros(n_seqs, max_blocks, dtype=torch.int64, device=device)
    for i in range(n_seqs):
        block_tables[i, :4] = torch.arange(i * 4 + 1, i * 4 + 5, device=device)
    seq_ids = torch.randint(0, n_seqs, (T,), generator=g, dtype=torch.int64).to(device)
    positions = torch.randint(0, 32, (T,), generator=g, dtype=torch.int64).to(device)
    return kv, latent, seq_ids, positions, block_tables


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_ref(device: str, dtype: torch.dtype) -> None:
    kv, latent, seq_ids, positions, block_tables = _make_case(device, dtype)
    kv_kernel = kv.clone()
    kv_ref = kv.clone()
    store_paged_kv(kv_kernel, latent, seq_ids, positions, block_tables)
    store_paged_kv_ref(kv_ref, latent, seq_ids, positions, block_tables)
    assert torch.equal(kv_kernel, kv_ref)
    # 只写了目标槽位，其余保持 0
    bs = kv.shape[1]
    blocks = block_tables[seq_ids, positions // bs]
    slots = positions % bs
    written = set(zip(blocks.tolist(), slots.tolist()))
    for b in range(kv.shape[0]):
        for s in range(bs):
            row = kv_kernel[b, s]
            if (b, s) not in written:
                assert torch.equal(row, torch.zeros_like(row))


@pytest.mark.parametrize("device", _devices())
def test_matches_naive_loop(device: str) -> None:
    kv, latent, seq_ids, positions, block_tables = _make_case(device, torch.float32, seed=7)
    store_paged_kv(kv, latent, seq_ids, positions, block_tables)
    bs = kv.shape[1]
    naive = torch.zeros_like(kv)
    for t in range(latent.shape[0]):
        b = int(block_tables[int(seq_ids[t]), int(positions[t]) // bs])
        s = int(positions[t]) % bs
        naive[b, s] = latent[t]
    assert torch.equal(kv, naive)


@pytest.mark.parametrize("device", _devices())
def test_empty_batch(device: str) -> None:
    kv = torch.zeros(10, 8, 576, device=device)
    latent = torch.zeros(0, 576, device=device)
    ids = torch.zeros(0, dtype=torch.int64, device=device)
    bt = torch.zeros(2, 4, dtype=torch.int64, device=device)
    store_paged_kv(kv, latent, ids, ids, bt)  # 不崩溃
    assert torch.equal(kv, torch.zeros_like(kv))

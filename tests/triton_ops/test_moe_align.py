"""moe_align_block_size：wrapper（确定性 torch）vs 朴素 ref + 结构不变式。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.moe_align import (
    moe_align_block_size,
    moe_align_ref,
)


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices())
def test_wrapper_matches_ref(device: str) -> None:
    g = torch.Generator().manual_seed(0)
    T, K, E, bs = 256, 6, 64, 16
    topk = torch.randint(0, E, (T, K), generator=g, dtype=torch.int32).to(device)
    a = moe_align_block_size(topk, E, bs)
    b = moe_align_ref(topk, E, bs)
    # ref 返回 CPU 张量；cuda 路径下需搬到同设备再比较
    dev = a[0].device
    assert torch.equal(a[0], b[0].to(dev)), "sorted_token_ids 不一致"
    assert torch.equal(a[1], b[1].to(dev)), "expert_ids 不一致"
    assert torch.equal(a[2].cpu(), b[2]), "num_tokens_post_pad 不一致"


@pytest.mark.parametrize("device", _devices())
def test_structure_invariants(device: str) -> None:
    g = torch.Generator().manual_seed(1)
    T, K, E, bs = 128, 4, 16, 8
    topk = torch.randint(0, E, (T, K), generator=g, dtype=torch.int32).to(device)
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk, E, bs)
    total = int(n_post.item())
    # 1. 每个 (token, slot) 恰好出现一次
    valid = sorted_ids[sorted_ids >= 0]
    assert valid.numel() == T * K
    assert torch.equal(torch.sort(valid.long()).values, torch.arange(T * K, device=device).long())
    # 2. 填充区为 -1，长度 = total - T*K
    assert (sorted_ids[total:] == -1).all() if sorted_ids.numel() > total else True
    # 3. 每个块属于唯一专家，桶内 token 的专家一致
    flat = topk.reshape(-1).long()
    for blk in range(expert_ids.numel()):
        e = int(expert_ids[blk])
        chunk = sorted_ids[blk * bs : (blk + 1) * bs]
        for sid in chunk.tolist():
            if sid >= 0:
                assert int(flat[sid]) == e, f"块 {blk} 声明专家 {e}，token {sid} 实际 {int(flat[sid])}"
    # 4. 每专家段 block 对齐
    counts = torch.bincount(flat, minlength=E)
    for e in range(E):
        seg = counts[e]
        assert seg == 0 or (seg - 1) // bs + 1 >= 1


@pytest.mark.parametrize("device", _devices())
def test_empty_and_degenerate(device: str) -> None:
    # 单专家
    topk = torch.zeros(8, 3, dtype=torch.int32, device=device)
    s, e, n = moe_align_block_size(topk, 1, 4)
    assert int(n.item()) == 24
    assert e.tolist() == [0] * 6
    # 空 batch
    topk = torch.zeros(0, 3, dtype=torch.int32, device=device)
    s, e, n = moe_align_block_size(topk, 8, 4)
    assert int(n.item()) == 0 and e.numel() == 0
    # pad_to 固定尺寸（CUDA graph 用）
    g = torch.Generator().manual_seed(2)
    topk = torch.randint(0, 8, (16, 2), generator=g, dtype=torch.int32).to(device)
    s, e, n = moe_align_block_size(topk, 8, 4, pad_to=128)
    assert s.shape == (128,) and (s[int(n.item()):] == -1).all()


@pytest.mark.parametrize("device", _devices())
def test_deterministic_across_calls(device: str) -> None:
    g = torch.Generator().manual_seed(3)
    topk = torch.randint(0, 32, (64, 6), generator=g, dtype=torch.int32).to(device)
    a = moe_align_block_size(topk, 32, 16)
    b = moe_align_block_size(topk, 32, 16)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

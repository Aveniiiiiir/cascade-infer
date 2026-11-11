"""apply_token_bitmask：kernel vs ref 对拍 + 掩码语义。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.triton_ops.apply_so_bitmask import (
    apply_token_bitmask,
    apply_token_bitmask_ref,
)


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_case(device: str, seed: int = 0, vocab: int = 4096):
    g = torch.Generator().manual_seed(seed)
    Q, rows = 16, 6
    logits = torch.randn(Q, vocab, generator=g, dtype=torch.float32).to(device)
    num_words = (vocab + 31) // 32
    words = torch.randint(0, 2**31, (rows, num_words), generator=g, dtype=torch.int64).to(torch.int32).to(device)
    words[0].fill_(-1)  # 全 1 行：全合法
    words[1].fill_(0)  # 全 0 行：全禁止
    row_map = torch.randint(0, rows, (Q,), generator=g, dtype=torch.int32).to(device)
    row_map[0] = 0
    row_map[1] = 1
    return logits, words, row_map, vocab


@pytest.mark.parametrize("device", _devices())
def test_kernel_matches_ref(device: str) -> None:
    logits, words, row_map, vocab = _make_case(device)
    a = logits.clone()
    b = logits.clone()
    apply_token_bitmask(a, words, row_map)
    apply_token_bitmask_ref(b, words, row_map)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", _devices())
def test_mask_semantics(device: str) -> None:
    logits, words, row_map, vocab = _make_case(device, seed=3)
    out = logits.clone()
    apply_token_bitmask(out, words, row_map)
    idx = torch.arange(vocab, device=device)
    w = idx // 32
    b = idx % 32
    for i in range(out.shape[0]):
        r = int(row_map[i])
        allowed = ((words[r].long()[w] >> b) & 1).bool()
        assert torch.isinf(out[i][~allowed]).all() and (out[i][~allowed] == float("-inf")).all()
        assert torch.equal(out[i][allowed], logits[i][allowed])
    # 行 0（全 1 掩码）原样；行 1（全 0）全 -inf
    assert torch.equal(out[0], logits[0])
    assert (out[1] == float("-inf")).all()


@pytest.mark.parametrize("device", _devices())
def test_identity_row_map(device: str) -> None:
    """普通场景 row_map = arange：逐行掩码。"""
    torch.manual_seed(1)
    vocab = 2048
    logits = torch.randn(8, vocab, device=device)
    num_words = (vocab + 31) // 32
    words = torch.randint(0, 2**31, (8, num_words), dtype=torch.int64).to(torch.int32).to(device)
    row_map = torch.arange(8, dtype=torch.int32, device=device)
    out = logits.clone()
    apply_token_bitmask(out, words, row_map)
    ref = logits.clone()
    apply_token_bitmask_ref(ref, words, row_map)
    assert torch.equal(out, ref)

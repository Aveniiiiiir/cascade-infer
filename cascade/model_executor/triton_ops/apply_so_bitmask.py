"""apply_token_bitmask：结构化输出的 vocab 位掩码施加（logits 原地 -inf）。

对应 DeepInfer tilelang_ops/structure_output.py 的
`apply_token_bitmask_by_step_inplace`。位掩码布局：int32
`[rows, ceil(vocab/32)]`，第 i 词的第 j 位为 1 表示 token (i*32+j) 合法。
`row_map` 把 logits 行映射到位掩码行（MTP 变长验证时一行对应多步掩码；
普通场景 row_map = arange）。
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_BLOCK_V = 1024


if triton is not None:

    @triton.jit
    def _apply_bitmask_kernel(
        logits_ptr,  # [Q, vocab] fp32 连续，原地写
        bitmask_ptr,  # [rows, num_words] int32 连续
        row_map_ptr,  # [Q] int32: logits 行 → 位掩码行
        vocab,
        num_words,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        vblk = tl.program_id(1)
        bm_row = tl.load(row_map_ptr + row)
        v = vblk * BLOCK_V + tl.arange(0, BLOCK_V)
        w = v // 32
        b = v % 32
        in_vocab = v < vocab
        word = tl.load(bitmask_ptr + bm_row * num_words + w, mask=in_vocab & (w < num_words), other=0)
        bit = (word >> b) & 1
        val = tl.load(logits_ptr + row * vocab + v, mask=in_vocab)
        out = tl.where(bit == 1, val, float("-inf"))
        tl.store(logits_ptr + row * vocab + v, out, mask=in_vocab)


def apply_token_bitmask(
    logits: torch.Tensor,  # [Q, vocab] fp32 连续（原地修改）
    bitmask: torch.Tensor,  # [rows, ceil(vocab/32)] int32 连续
    row_map: torch.Tensor,  # [Q] int32
) -> None:
    assert logits.ndim == 2 and logits.is_contiguous()
    assert bitmask.is_contiguous() and bitmask.dtype == torch.int32
    Q, vocab = logits.shape
    if Q == 0:
        return
    if row_map is None:
        row_map = torch.arange(Q, dtype=torch.int32, device=logits.device)
    assert row_map.dtype == torch.int32 and row_map.shape == (Q,)
    if use_triton(logits.device):
        grid = (Q, triton.cdiv(vocab, _BLOCK_V))
        _apply_bitmask_kernel[grid](
            logits, bitmask, row_map, vocab, bitmask.shape[1], BLOCK_V=_BLOCK_V
        )
    else:
        apply_token_bitmask_ref(logits, bitmask, row_map)


def apply_token_bitmask_ref(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    row_map: torch.Tensor,
) -> None:
    """PyTorch 参考实现（向量化位展开）。"""
    Q, vocab = logits.shape
    rows, num_words = bitmask.shape
    assert num_words * 32 >= vocab
    idx = torch.arange(vocab, device=logits.device)
    w = idx // 32
    b = idx % 32
    # [rows, vocab] 合法位
    words = bitmask.to(torch.int64)
    allowed = ((words[:, w] >> b) & 1).bool()  # [rows, vocab]
    mask = ~allowed[row_map.long()]  # [Q, vocab]
    logits.masked_fill_(mask, float("-inf"))


__all__ = ["apply_token_bitmask", "apply_token_bitmask_ref"]

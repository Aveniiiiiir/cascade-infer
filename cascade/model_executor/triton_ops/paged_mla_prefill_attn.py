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

两种 KV 布局（与 decode kernel 同构）：bf16 单张量 / fp8（c + rope + scale，
kernel 内 dequant）。`uniform_v` 路径（全段等长）的元数据全部由 device
张量构造，是 CUDA graph 多 token（投机验证批）回放的唯一 graph-safe 路径。
"""

from __future__ import annotations

import torch

from cascade.model_executor.triton_ops.ops import tl, triton, use_triton

_DEFAULT_BLOCK_M = 16
_DEFAULT_BLOCK_N = 32


def _fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


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

    @triton.jit
    def _prefill_attn_kernel_fp8(
        q_ptr,  # [T, H, D] bf16
        c_ptr,  # [num_blocks, BS, KV_LORA] float8_e4m3fn
        rope_ptr,  # [num_blocks, BS, ROPE] bf16
        sc_ptr,  # [num_blocks, BS] fp32
        block_tables_ptr,  # [num_seqs, MAX_BLOCKS] int32
        qo_seg_ptr,
        qo_qbase_ptr,
        qo_pos_ptr,
        qo_len_ptr,
        seqlens_ptr,
        out_ptr,  # [T, H, KV_LORA] fp32
        H,
        MAX_BLOCKS,
        SCALE,
        BS: tl.constexpr,
        KV_LORA: tl.constexpr,
        ROPE: tl.constexpr,
        ROPE_PAD: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        D: tl.constexpr = KV_LORA + ROPE
        GROUP: tl.constexpr = KV_LORA // NUM_GROUPS
        seg = tl.load(qo_seg_ptr + pid_m)
        qbase = tl.load(qo_qbase_ptr + pid_m)
        q_pos0 = tl.load(qo_pos_ptr + pid_m)
        q_len = tl.load(qo_len_ptr + pid_m)
        seqlen = tl.load(seqlens_ptr + seg)
        kv_end = tl.minimum(q_pos0 + q_len, seqlen)
        offs_m = tl.arange(0, BLOCK_M)
        m_valid = offs_m < q_len
        offs_d = tl.arange(0, KV_LORA)
        offs_r = tl.arange(0, ROPE_PAD)
        rope_valid = offs_r < ROPE
        q_row = qbase + offs_m
        q_base_p = q_ptr + q_row[:, None].to(tl.int64) * (H * D) + h * D
        q_abs = tl.load(q_base_p + offs_d[None, :], mask=m_valid[:, None], other=0.0)
        q_rope = tl.load(q_base_p + KV_LORA + offs_r[None, :], mask=m_valid[:, None] & rope_valid[None, :], other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, KV_LORA], dtype=tl.float32)
        pos_q = q_pos0 + offs_m
        for t0 in range(0, kv_end, BLOCK_N):
            tok = t0 + tl.arange(0, BLOCK_N)
            valid = tok < kv_end
            blk = tl.load(
                block_tables_ptr + seg * MAX_BLOCKS + tok // BS, mask=valid, other=0
            ).to(tl.int64)
            flat = blk * BS + (tok % BS)
            c_raw = tl.load(
                c_ptr + flat[:, None] * KV_LORA + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            offs_g = tl.arange(0, NUM_GROUPS)
            sc = tl.load(
                sc_ptr + flat[:, None] * NUM_GROUPS + offs_g[None, :],
                mask=valid[:, None],
                other=1.0,
            )  # [BLOCK_N, NUM_GROUPS]
            c3 = tl.reshape(c_raw.to(tl.float32), [BLOCK_N, NUM_GROUPS, GROUP])
            c = tl.reshape(c3 * sc[:, :, None], [BLOCK_N, KV_LORA]).to(tl.bfloat16)
            rope = tl.load(
                rope_ptr + flat[:, None] * ROPE + offs_r[None, :],
                mask=valid[:, None] & rope_valid[None, :],
                other=0.0,
            )
            s = tl.dot(q_abs, tl.trans(c)) + tl.dot(q_rope, tl.trans(rope))
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
    kv: torch.Tensor,  # [num_blocks, BS, D] bf16；fp8 时为 c [.., KV_LORA] fp8
    block_tables: torch.Tensor,  # [num_seqs, max_blocks] int32/int64
    seqlens: torch.Tensor,  # [num_seqs] int64：各 seq 的 KV 总数（含历史 chunk）
    segments: list[tuple[int, int, int]],  # (seq 行, 段首绝对位置, 段长)
    scale: float,
    kv_lora: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    uniform_v: int | None = None,  # 等长验证批：段元数据全张量化（图捕获安全）
    rope: torch.Tensor | None = None,  # fp8：k_rope [num_blocks, BS, ROPE] bf16
    kv_scale: torch.Tensor | None = None,  # fp8：每 token scale [num_blocks, BS] fp32
) -> torch.Tensor:
    """→ 加权 latent [T, H, kv_lora] fp32（行序与 q 一致）。

    ``uniform_v``：所有段等长（= v，行数 n_seqs×v）。此时**忽略 segments**，
    q 块元数据（qo_seg/qbase/pos/len）全部由 device 张量构造——位置从
    ``seqlens − v`` 推导（seqlens 来自 positions[::v]+v，图捕获期由回放
    写入的 positions 缓冲驱动）。这是 CUDA graph 多 token 解码的唯一
    graph-safe 路径（eager 混合批仍走 segments + .item()）。
    """
    assert q.ndim == 3 and kv.ndim == 3
    T, H, D = q.shape
    BS = kv.shape[1]
    fp8 = kv.dtype == _fp8_dtype()
    num_groups = 1
    if fp8:
        assert rope is not None and kv_scale is not None, "fp8 KV 需要 rope/kv_scale"
        assert kv.shape[2] == kv_lora and rope.shape == (kv.shape[0], BS, D - kv_lora)
        assert kv_scale.shape[:2] == (kv.shape[0], BS)
        num_groups = kv_scale.shape[-1]
        assert kv_lora % num_groups == 0
        assert num_groups & (num_groups - 1) == 0, "分组数必须为 2 的幂（Triton reshape）"
    else:
        assert kv.shape[2] == D
    assert D - kv_lora > 0
    # BLOCK_N 下限 16（tl.dot 的 N 维约束）；块 id 逐 token 查表后不再要求
    # BLOCK_N 整除 KV block size
    block_n = max(16, block_n)
    rope_pad = max(16, triton.next_power_of_2(D - kv_lora))
    assert kv.is_contiguous() and block_tables.is_contiguous()
    # kernel 路径要求 tensor-core dtype；fp32（紧容差测试场景）走 ref
    if not (use_triton(q.device) and q.dtype in (torch.bfloat16, torch.float16)):
        # ref 路径：uniform_v 还原成显式 segments（eager 可同步，图捕获不会
        # 走到这里——图只在 CUDA+bf16 下构建）
        if uniform_v is not None:
            first = (seqlens - uniform_v).tolist()
            segments = [(i, int(first[i]), uniform_v) for i in range(len(first))]
        return paged_mla_prefill_attn_ref(
            q, kv, block_tables, seqlens, segments, scale, kv_lora, rope, kv_scale
        )
    assert q.is_contiguous()
    device = q.device
    if uniform_v is not None:
        # ---- 等长段：全张量化元数据（含 qo_pos 的 device 侧推导）----
        return _uniform_prefill(
            q, kv, block_tables, seqlens, uniform_v, scale, kv_lora,
            block_m=block_m, block_n=block_n, rope_pad=rope_pad, BS=BS, H=H, D=D,
            fp8=fp8, rope=rope, kv_scale=kv_scale, num_groups=num_groups,
        )
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
    _launch_prefill(
        q, kv, block_tables, seqlens, out,
        torch.tensor(seg_ids, dtype=torch.int32, device=device),
        torch.tensor(qbases, dtype=torch.int32, device=device),
        torch.tensor(poss, dtype=torch.int64, device=device),
        torch.tensor(lens, dtype=torch.int32, device=device),
        len(qbases), scale, kv_lora, rope_pad,
        block_m=block_m, block_n=block_n, BS=BS, H=H, D=D,
        fp8=fp8, rope=rope, kv_scale=kv_scale, num_groups=num_groups,
    )
    return out


def _launch_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    out: torch.Tensor,
    qo_seg: torch.Tensor,
    qo_qbase: torch.Tensor,
    qo_pos: torch.Tensor,
    qo_len: torch.Tensor,
    num_q_blocks: int,
    scale: float,
    kv_lora: int,
    rope_pad: int,
    *,
    block_m: int,
    block_n: int,
    BS: int,
    H: int,
    D: int,
    fp8: bool,
    rope: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
    num_groups: int,
) -> None:
    common = dict(
        BS=BS, KV_LORA=kv_lora, ROPE=D - kv_lora, ROPE_PAD=rope_pad,
        BLOCK_M=block_m, BLOCK_N=block_n, num_warps=8,
    )
    bt = block_tables.to(torch.int32)
    if fp8:
        _prefill_attn_kernel_fp8[(num_q_blocks, H)](
            q, kv, rope, kv_scale, bt, qo_seg, qo_qbase, qo_pos, qo_len, seqlens, out,
            H, block_tables.shape[1], scale, NUM_GROUPS=num_groups, **common,
        )
    else:
        _prefill_attn_kernel[(num_q_blocks, H)](
            q, kv, bt, qo_seg, qo_qbase, qo_pos, qo_len, seqlens, out,
            H, block_tables.shape[1], scale, **common,
        )


def _uniform_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    v: int,
    scale: float,
    kv_lora: int,
    *,
    block_m: int,
    block_n: int,
    rope_pad: int,
    BS: int,
    H: int,
    D: int,
    fp8: bool,
    rope: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
    num_groups: int,
) -> torch.Tensor:
    """等长段（每 seq v 行）的图安全 prefill：元数据全为 device 张量，
    位置由 ``seqlens − v`` 推导（回放期 positions 缓冲变化 ⇒ qo_pos 自动更新）。"""
    device = q.device
    T = q.shape[0]
    n = int(seqlens.shape[0])
    assert n * v == T, (n, v, T)
    nb = (v + block_m - 1) // block_m
    blk = torch.arange(n * nb, dtype=torch.int64, device=device)
    seg_i = blk // nb
    blk_j = blk % nb
    qo_seg = seg_i.to(torch.int32)
    qo_qbase = (seg_i * v + blk_j * block_m).to(torch.int32)
    qo_pos = (seqlens - v).to(torch.int64).repeat_interleave(nb) + blk_j * block_m
    qo_len = torch.clamp(v - blk_j * block_m, max=block_m).to(torch.int32)
    out = torch.empty(T, H, kv_lora, dtype=torch.float32, device=device)
    _launch_prefill(
        q, kv, block_tables, seqlens, out,
        qo_seg, qo_qbase, qo_pos, qo_len,
        n * nb, scale, kv_lora, rope_pad,
        block_m=block_m, block_n=block_n, BS=BS, H=H, D=D,
        fp8=fp8, rope=rope, kv_scale=kv_scale, num_groups=num_groups,
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
    rope: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch 参考实现（逐段 + 因果掩码；fp8 时先 dequant 成 bf16 行）。"""
    T, H, D = q.shape
    BS = kv.shape[1]
    fp8 = kv.dtype == _fp8_dtype()
    out = torch.empty(T, H, kv_lora, dtype=torch.float32, device=q.device)
    base = 0
    for row, pos_start, seg_len in segments:
        P = min(pos_start + seg_len, int(seqlens[row]))
        nblocks = (P + BS - 1) // BS
        rows_idx = block_tables[row, :nblocks].long()
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
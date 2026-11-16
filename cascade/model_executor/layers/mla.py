"""MLA attention layer (DeepSeek V2 family).

两条前向路径，数学上等价：

1. **显式路径**（`_attend_segment_explicit`）：从 cache 取回 latent c，用
   `kv_b_proj` 材料化每头 k_nope/v —— 与 HF transformers 的非 MLA-cache 路径
   逐步同构，是集群 e2e 对拍 HF 的正确性 oracle。
2. **吸收路径**（`_attend_segment_absorbed`）：q_nope 吸收 W_UK（q_abs 与
   latent c 点积）、输出经 W_UV 上投影 —— KV 只需 latent 576 维，paged KV
   布局由此设计；是 Triton decode kernel 的语义规范。

KV cache 存储内容：`[c(normed) ‖ k_rope(已rope)]`，每 token 每层 576 维。
本层把写入与注意力解耦：先 store 全部新 token 的 latent，再按 seq 段注意力
（段内 causal，段间独立）—— chunked prefill 天然正确（新 chunk 的 q 对
[0, last] 全历史可见，mask 由绝对位置决定）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cascade.config import ModelConfig
from cascade.model_executor.layers.layernorm import RMSNorm
from cascade.model_executor.layers.linear import Linear
from cascade.model_executor.layers.rotary import RotaryEmbedding
from cascade.worker.cache_engine import CacheEngine


class MLAAttention(nn.Module):
    """HF 权重命名（q_a_proj / kv_a_proj_with_mqa / ...），加载器零映射。"""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        h = cfg.num_attention_heads
        self.num_heads = h
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.v_head = cfg.v_head_dim
        self.kv_lora = cfg.kv_lora_rank

        if cfg.q_lora_rank:
            self.q_a_proj = Linear(cfg.hidden_size, cfg.q_lora_rank)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
            self.q_b_proj = Linear(cfg.q_lora_rank, h * (self.qk_nope + self.qk_rope))
        else:
            self.q_proj = Linear(cfg.hidden_size, h * (self.qk_nope + self.qk_rope))

        self.kv_a_proj_with_mqa = Linear(cfg.hidden_size, self.kv_lora + self.qk_rope)
        self.kv_a_layernorm = RMSNorm(self.kv_lora, eps=cfg.rms_norm_eps)
        self.kv_b_proj = Linear(self.kv_lora, h * (self.qk_nope + self.v_head))
        self.o_proj = Linear(h * self.v_head, cfg.hidden_size)
        self.scale = cfg.softmax_scale

    # ------------------------------------------------------------------ projections

    def project_q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T,H] → (q_nope [T,h,128], q_rope [T,h,64])（未 rope）。"""
        if self.cfg.q_lora_rank:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        T = x.shape[0]
        q = q.view(T, self.num_heads, self.qk_nope + self.qk_rope)
        return q[..., : self.qk_nope], q[..., self.qk_nope :]

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T,H] → (c_normed [T,512], k_rope [T,64])（未 rope）。

        c 必须 norm 后入 cache：吸收路径 k_nope = U_k @ c 用的就是 normed c。
        """
        kv = self.kv_a_proj_with_mqa(x)
        c = self.kv_a_layernorm(kv[..., : self.kv_lora])
        k_rope = kv[..., self.kv_lora :]
        return c, k_rope

    # ------------------------------------------------------------------ absorbed weight views

    @property
    def u_k(self) -> torch.Tensor:
        """kv_b_proj 权重的 k 部分: [h, 128, 512]（每头 W_UK，k_nope = U_k @ c）。"""
        w = self.kv_b_proj.weight  # [h*(128+128), 512]
        return w.view(self.num_heads, self.qk_nope + self.v_head, self.kv_lora)[
            :, : self.qk_nope, :
        ]

    @property
    def u_v(self) -> torch.Tensor:
        """kv_b_proj 权重的 v 部分: [h, 128, 512]（每头 W_UV，v = U_v @ c）。"""
        w = self.kv_b_proj.weight
        return w.view(self.num_heads, self.qk_nope + self.v_head, self.kv_lora)[
            :, self.qk_nope :, :
        ]

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        x: torch.Tensor,  # [T, H] 段拼接
        positions: torch.Tensor,  # [T] 绝对位置
        seq_ids: torch.Tensor,  # [T]
        segment_lens: list[int],  # 每 seq 段的 token 数
        block_tables: torch.Tensor,  # [n_seqs, max_blocks]
        cache: CacheEngine,
        rope: RotaryEmbedding,
        use_absorbed: bool = False,
        num_splits: int | None = None,  # 图捕获时固定（避免 .item() 同步）
        rope_ensure: bool = True,  # False = 调用方已预热 rope 缓冲（免同步热路径）
    ) -> torch.Tensor:
        # 1. 投影 + rope
        q_nope, q_rope = self.project_q(x)
        c, k_rope = self.project_kv(x)
        q_rope, k_rope = rope.apply(q_rope, k_rope, positions, ensure=rope_ensure)
        # 2. 写 paged KV：[c ‖ k_rope(已 rope)]（triton/ref 自动分发）
        latent = torch.cat((c, k_rope), dim=-1)  # [T, 576]
        cache.store_latent(self.layer_idx, latent, seq_ids, positions, block_tables)
        # 3. 注意力
        if use_absorbed:
            out = self._forward_absorbed_batched(
                q_nope, q_rope, positions, seq_ids, segment_lens, block_tables, cache, num_splits
            )
        else:
            out = self._forward_explicit_loop(
                q_nope, q_rope, positions, seq_ids, segment_lens, block_tables, cache
            )
        return self.o_proj(out)  # [T, h*128] → [T, H]

    def _forward_explicit_loop(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        segment_lens: list[int],
        block_tables: torch.Tensor,
        cache: CacheEngine,
    ) -> torch.Tensor:
        """显式路径（HF 对拍 oracle）：逐段材料化 k_nope/v。"""
        outs = []
        offset = 0
        for seg_i, seg_len in enumerate(segment_lens):
            sl = slice(offset, offset + seg_len)
            out = self._attend_segment_explicit(
                q_nope[sl], q_rope[sl], positions[sl],
                cache.gather_range_reference(
                    self.layer_idx,
                    int(seq_ids[sl.start].item()),
                    int(positions[sl.start].item()) + seg_len,
                    block_tables,
                ),
            )
            outs.append(out)
            offset += seg_len
        return torch.cat(outs, dim=0)

    def _forward_absorbed_batched(
        self,
        q_nope: torch.Tensor,  # [T, h, 128]
        q_rope: torch.Tensor,  # [T, h, 64]（已 rope）
        positions: torch.Tensor,  # [T]
        seq_ids: torch.Tensor,  # [T]
        segment_lens: list[int],
        block_tables: torch.Tensor,  # [n_seqs, max_blocks]
        cache: CacheEngine,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        """吸收路径批处理：多 token 段走 prefill kernel，单 token 段（decode）
        走 split-K decode kernel；kernel 按精度自动分发（fp32 → ref）。

        段与 block_tables 行 1:1（一个 seq 每步至多一个段）。
        **图捕获安全**：纯 decode 批（segment_lens 全 1）走无 .item() 的
        张量化路径（seqlens = positions + 1），num_splits 由图 runner 固定。
        """
        from cascade.model_executor.triton_ops.paged_mla_decode_attn import (
            paged_mla_decode_attn,
        )
        from cascade.model_executor.triton_ops.paged_mla_prefill_attn import (
            paged_mla_prefill_attn,
        )

        T = q_nope.shape[0]
        kv = cache.layer_view(self.layer_idx)
        # q 吸收 W_UK：q_abs 与 latent c 点积（fp32 计算）
        q_abs = torch.einsum("thd,hdk->thk", q_nope.float(), self.u_k.float())
        q_full = torch.cat((q_abs, q_rope.float()), dim=-1).to(kv.dtype)  # [T, h, 576]
        all_decode = all(n == 1 for n in segment_lens)
        out_latent = torch.empty(T, self.num_heads, self.kv_lora, dtype=torch.float32, device=kv.device)
        if all_decode:
            # ---- 纯 decode 批：图捕获安全（无 .item()）----
            seqlens = positions.to(torch.int64) + 1
            out_d = paged_mla_decode_attn(
                q_full, kv, block_tables, seqlens, self.scale, self.kv_lora, num_splits=num_splits
            )
            out_latent = out_d.to(torch.float32)
        else:
            # ---- 混合批（eager）：段偏移与每段 KV 上界 ----
            offsets = []
            acc = 0
            for n in segment_lens:
                offsets.append(acc)
                acc += n
            assert acc == T
            seqlens = torch.tensor(
                [int(positions[o].item()) + n for o, n in zip(offsets, segment_lens)],
                dtype=torch.int64,
                device=kv.device,
            )
            # 多 token 段：prefill kernel（varlen 因果）
            multi_segs = [
                (i, int(positions[o].item()), n)
                for i, (o, n) in enumerate(zip(offsets, segment_lens))
                if n > 1
            ]
            if multi_segs:
                rows = torch.cat(
                    [torch.arange(offsets[i], offsets[i] + n, device=kv.device) for i, _, n in multi_segs]
                )
                q_m = q_full.index_select(0, rows).contiguous()
                out_latent[rows] = paged_mla_prefill_attn(
                    q_m, kv, block_tables, seqlens, multi_segs, self.scale, self.kv_lora
                )
            # 单 token 段（decode）：split-K kernel
            decode_segs = [i for i, n in enumerate(segment_lens) if n == 1]
            if decode_segs:
                rows = torch.tensor([offsets[i] for i in decode_segs], device=kv.device)
                q_d = q_full.index_select(0, rows).contiguous()
                bt_d = block_tables.index_select(0, torch.tensor(decode_segs, device=kv.device))
                sl_d = seqlens.index_select(0, torch.tensor(decode_segs, device=kv.device))
                out_d = paged_mla_decode_attn(
                    q_d, kv, bt_d, sl_d, self.scale, self.kv_lora, num_splits=num_splits
                )
                out_latent[rows] = out_d.to(torch.float32)
        # W_UV 上投影
        out = torch.einsum("thk,hdk->thd", out_latent, self.u_v.float())  # [T, h, 128] fp32
        return out.to(kv.dtype).reshape(T, self.num_heads * self.v_head)

    # ------------------------------------------------------------------ explicit (HF oracle)

    def _attend_segment_explicit(
        self,
        q_nope: torch.Tensor,  # [L, h, 128]
        q_rope: torch.Tensor,  # [L, h, 64]
        positions: torch.Tensor,  # [L]
        kv: torch.Tensor,  # [P, 576]（c ‖ k_rope 已 rope）
    ) -> torch.Tensor:
        P = kv.shape[0]
        c_kv = kv[..., : self.kv_lora]  # [P, 512]
        k_rope_kv = kv[..., self.kv_lora :]  # [P, 64]
        # 材料化每头 k_nope / v（与 HF 非缓存路径同构）
        kv_b = self.kv_b_proj(c_kv).view(P, self.num_heads, self.qk_nope + self.v_head)
        k_nope = kv_b[..., : self.qk_nope]  # [P, h, 128]
        v = kv_b[..., self.qk_nope :]  # [P, h, 128]
        q_full = torch.cat((q_nope, q_rope), dim=-1)  # [L, h, 192]
        k_full = torch.cat((k_nope, k_rope_kv.unsqueeze(1).expand(P, self.num_heads, self.qk_rope)), dim=-1)
        # 数值路径与 HF 逐算子对齐：4D 批量 matmul（[1, h, L, D] ×
        # [1, h, D, P]）→ fp32 softmax → 转回 bf16 → bf16 matmul。
        # einsum 的 3D bmm 会选到不同的 cuBLAS kernel/累加顺序，与 HF 的
        # 4D matmul 有可观测差异（layer0 输出 max|d| ≈ 0.4）。
        q4 = q_full.transpose(0, 1).unsqueeze(0)  # [1, h, L, D]
        k4 = k_full.transpose(0, 1).unsqueeze(0)  # [1, h, P, D]
        v4 = v.transpose(0, 1).unsqueeze(0)  # [1, h, P, Dv]
        scores = torch.matmul(q4, k4.transpose(2, 3)) * self.scale
        # mask[q_pos, k_pos] = positions[q] >= k：unsqueeze(1) 使 positions 成为
        # 列向量 [L,1]，与 arange(P) 行向量 [1,P] 广播为 [L,P]（此前
        # unsqueeze(0) 得到的是转置的 [P,L]，L==P 时掩码方向错误）
        mask = positions.unsqueeze(1) >= torch.arange(P, device=positions.device).unsqueeze(0)  # [L,P]
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q4.dtype)
        out = torch.matmul(w, v4).squeeze(0).transpose(0, 1)  # [L, h, Dv]
        L = q_nope.shape[0]
        return out.reshape(L, self.num_heads * self.v_head)

    # ------------------------------------------------------------------ absorbed (kernel spec)

    def _attend_segment_absorbed(
        self,
        q_nope: torch.Tensor,  # [L, h, 128]
        q_rope: torch.Tensor,  # [L, h, 64]
        positions: torch.Tensor,  # [L]
        kv: torch.Tensor,  # [P, 576]
    ) -> torch.Tensor:
        P = kv.shape[0]
        # q_nope 吸收 W_UK: [L,h,128] × [h,128,512] → [L,h,512]
        q_abs = torch.einsum("lhd,hdk->lhk", q_nope.float(), self.u_k.float())
        q_full = torch.cat((q_abs, q_rope.float()), dim=-1)  # [L, h, 576]
        scores = torch.einsum("lhk,pk->hlp", q_full, kv.float()) * self.scale
        # 同上：mask[q, k] = positions[q] >= k，unsqueeze(1) 得 [L,P]（非转置）
        mask = positions.unsqueeze(1) >= torch.arange(P, device=positions.device).unsqueeze(0)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        # 加权累积 latent（各头独立权重），再 W_UV 上投影
        out_latent = torch.einsum("hlp,pk->lhk", w, kv[..., : self.kv_lora].float())  # [L,h,512]
        out = torch.einsum("lhk,hdk->lhd", out_latent, self.u_v.float())  # [L,h,128]
        L = q_nope.shape[0]
        return out.reshape(L, self.num_heads * self.v_head).to(q_nope.dtype)


__all__ = ["MLAAttention"]

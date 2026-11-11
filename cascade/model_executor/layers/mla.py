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
    ) -> torch.Tensor:
        # 1. 投影 + rope
        q_nope, q_rope = self.project_q(x)
        c, k_rope = self.project_kv(x)
        q_rope, k_rope = rope.apply(q_rope, k_rope, positions)
        # 2. 写 paged KV：[c ‖ k_rope(已 rope)]（triton/ref 自动分发）
        latent = torch.cat((c, k_rope), dim=-1)  # [T, 576]
        cache.store_latent(self.layer_idx, latent, seq_ids, positions, block_tables)
        # 3. 分段注意力
        outs = []
        offset = 0
        for seg_i, seg_len in enumerate(segment_lens):
            sl = slice(offset, offset + seg_len)
            out = self._attend_segment(
                q_nope[sl], q_rope[sl], positions[sl],
                seq_row=int(seq_ids[sl.start].item()),
                num_kv=int(positions[sl.start].item()) + seg_len,  # 段首位置 + 段长 = 历史总数
                block_tables=block_tables,
                cache=cache,
                use_absorbed=use_absorbed,
            )
            outs.append(out)
            offset += seg_len
        out = torch.cat(outs, dim=0)  # [T, h*128]
        return self.o_proj(out)

    def _attend_segment(
        self,
        q_nope: torch.Tensor,  # [L, h, 128]
        q_rope: torch.Tensor,  # [L, h, 64]（已 rope）
        positions: torch.Tensor,  # [L]（绝对）
        seq_row: int,
        num_kv: int,
        block_tables: torch.Tensor,
        cache: CacheEngine,
        use_absorbed: bool,
    ) -> torch.Tensor:
        kv = cache.gather_range_reference(self.layer_idx, seq_row, num_kv, block_tables)  # [P, 576]
        if use_absorbed:
            return self._attend_segment_absorbed(q_nope, q_rope, positions, kv)
        return self._attend_segment_explicit(q_nope, q_rope, positions, kv)

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
        # fp32 分数 + 位置 causal 掩码
        scores = torch.einsum("lhd,phd->hlp", q_full.float(), k_full.float()) * self.scale
        mask = positions.unsqueeze(0) >= torch.arange(P, device=positions.device).unsqueeze(1)  # [L,P]
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = torch.einsum("hlp,phd->lhd", w, v.float()).to(v.dtype)  # [L, h, 128]
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
        mask = positions.unsqueeze(0) >= torch.arange(P, device=positions.device).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        # 加权累积 latent（各头独立权重），再 W_UV 上投影
        out_latent = torch.einsum("hlp,pk->lhk", w, kv[..., : self.kv_lora].float())  # [L,h,512]
        out = torch.einsum("lhk,hdk->lhd", out_latent, self.u_v.float())  # [L,h,128]
        L = q_nope.shape[0]
        return out.reshape(L, self.num_heads * self.v_head).to(q_nope.dtype)


__all__ = ["MLAAttention"]

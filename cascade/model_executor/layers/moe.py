"""MoE 层（DeepSeek V2 家族，noaux_tc sigmoid 路由）与 dense MLP。

专家权重以**堆叠张量**持有（w1/w3: [E, N, H]，w2: [E, H, N]）——分组
GEMM kernel 按专家索引的要求（vLLM 同构）；HF checkpoint 的逐专家键
（mlp.experts.E.gate_proj.weight）由加载器重映射到堆叠切片。

fused 路径经 triton_ops.fused_experts（moe_align + 双分组 GEMM，全程无
原子，浮点归约顺序固定）；无 triton/fp32 场景自动走逐专家参考路径。

路由语义：
- noaux_tc: 选取 top-k 用 scores + e_score_correction_bias（bias 只影响选择），
  权重取原始 scores；norm_topk_prob 时归一化；× routed_scaling_factor。
- n_group > 0（V2.5/V3）：先组内选 topk_group 个组，再组外 top-k。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.config import ModelConfig
from cascade.model_executor.triton_ops.fused_moe import fused_experts


class MLP(nn.Module):
    """SwiGLU FFN（HF 命名：gate_proj / up_proj / down_proj）。"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """路由门（HF 命名：weight / e_score_correction_bias）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.topk = cfg.num_experts_per_tok
        self.n_routed_experts = cfg.n_routed_experts
        self.weight = nn.Parameter(torch.empty((cfg.n_routed_experts, cfg.hidden_size)))
        nn.init.normal_(self.weight, std=0.02)
        if cfg.has_e_score_correction_bias:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))
        else:
            self.e_score_correction_bias = None
        self.n_group = 0  # V2-Lite: 0; V3 由 from_hf 覆盖（deepseek_v3.py）
        self.topk_group = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T,H] → (topk_idx [T,k], topk_weight [T,k])（fp32 权重）。"""
        logits = x @ self.weight.T  # [T, E]
        if self.cfg.scoring_func == "sigmoid":
            scores = logits.float().sigmoid()
        else:
            scores = logits.float().softmax(dim=-1)
        if self.cfg.has_e_score_correction_bias:
            scores_for_choice = scores + self.e_score_correction_bias.float().unsqueeze(0)
        else:
            scores_for_choice = scores
        if self.n_group > 0:
            # 组内选组：每组取 max，再选 topk_group 个组；未选中组的 token 出局
            group_size = self.n_routed_experts // self.n_group
            view = scores_for_choice.view(-1, self.n_group, group_size)
            group_scores = view.max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(view).scatter_(
                1, group_idx.unsqueeze(-1).expand(-1, -1, group_size), 1.0
            )
            scores_for_choice = view * group_mask
            scores_for_choice = scores_for_choice.view(-1, self.n_routed_experts)
        topk_idx = torch.topk(scores_for_choice, k=self.topk, dim=-1, sorted=False)[1]  # [T,k]
        topk_weight = scores.gather(1, topk_idx)  # 权重取原始 scores（无 bias）
        if self.cfg.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight * self.cfg.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekMoE(nn.Module):
    """堆叠专家权重（加载器把 HF 的 mlp.experts.E.* 映射到 w1/w3/w2 切片）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg)
        E, N, H = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.w1 = nn.Parameter(torch.empty(E, N, H))
        self.w3 = nn.Parameter(torch.empty(E, N, H))
        self.w2 = nn.Parameter(torch.empty(E, H, N))
        for p in (self.w1, self.w3, self.w2):
            nn.init.normal_(p, std=0.02)
        self.shared_experts = MLP(cfg.hidden_size, cfg.shared_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        topk_idx, topk_weight = self.gate(x)  # [T,k] int64 / fp32
        routed = fused_experts(
            x, self.w1, self.w3, self.w2, topk_weight, topk_ids=topk_idx.to(torch.int32)
        )
        return routed + self.shared_experts(x)

    # ------------------------------------------------------------------ 参考/调试

    def expert_ffn_ref(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        """单专家 FFN（参考语义文档）。"""
        return (F.silu(x @ self.w1[expert].T) * (x @ self.w3[expert].T)) @ self.w2[expert].T


__all__ = ["MLP", "MoEGate", "DeepseekMoE"]

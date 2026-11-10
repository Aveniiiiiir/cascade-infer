"""MoE 层（DeepSeek V2 家族，noaux_tc sigmoid 路由）与 dense MLP。

参考实现（逐专家循环）：正确性优先，fused Triton 路径（moe_align +
fused_moe_gemm）Phase 2 后期替换。路由语义：

- noaux_tc: 选取 top-k 用 scores + e_score_correction_bias（bias 只影响选择），
  权重取原始 scores；norm_topk_prob 时归一化；× routed_scaling_factor。
- n_group > 0（V2.5/V3）：先组内选 topk_group 个组，再组外 top-k。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.config import ModelConfig


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
            group_scores = scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)).scatter_(1, group_idx.unsqueeze(-1).expand(-1, -1, self.n_routed_experts // self.n_group), 1.0)
            scores_for_choice = scores_for_choice.view(-1, self.n_routed_experts) * group_mask.view(-1, self.n_routed_experts)
        topk_idx = torch.topk(scores_for_choice, k=self.topk, dim=-1, sorted=False)[1]  # [T,k]
        topk_weight = scores.gather(1, topk_idx)  # 权重取原始 scores（无 bias）
        if self.cfg.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight * self.cfg.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekMoE(nn.Module):
    """HF 命名：gate / experts.E / shared_experts —— 加载器零映射。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg)
        self.experts = nn.ModuleList(
            [MLP(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)]
        )
        self.shared_experts = MLP(cfg.hidden_size, cfg.shared_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        topk_idx, topk_weight = self.gate(x)  # [T,k]
        out = torch.zeros_like(x)
        # 参考路径：逐专家循环（正确、可读；fused 路径 Phase 2 后期）
        flat_expert = topk_idx.flatten()  # [T*k]
        for e in range(self.cfg.n_routed_experts):
            token_rows, k_slots = torch.nonzero(flat_expert == e, as_tuple=True)
            if token_rows.numel() == 0:
                continue
            rows = torch.unique(token_rows)
            weights = torch.zeros(T, device=x.device, dtype=topk_weight.dtype)
            weights[token_rows] = topk_weight[token_rows, k_slots]
            expert_out = self.experts[e](x[rows])
            out.index_add_(
                0, rows,
                expert_out.to(x.dtype) * weights[rows].to(x.dtype).unsqueeze(-1),
            )
        out = out + self.shared_experts(x)
        return out


__all__ = ["MLP", "MoEGate", "DeepseekMoE"]

"""MTP nextn 草案模型（DeepSeek-V3 语义）+ 测试用 mock。

V3 nextn 层 = checkpoint 的第 num_hidden_layers 层（num_nextn_predict_layers=1）：
- embed_tokens / lm_head / 层内 RMSNorm 与主模型共享
- eh_proj: [hidden, 2*hidden]（cat(hidden_prev, embed(token)) → hidden）
- draft 步自回归：hidden_j+1, logits_j+1 = layer(eh_proj(cat(h_j, emb(d_j)))

V2-Lite 无 nextn 权重 → MockDraftModel（确定性 f(seq_seed, 位置)，与
tests/cpu/mock_model 同构）驱动管线测试 —— 贪心等价性不受 mock 接受率
影响（管线 bug 必然破坏逐 token 一致）。

引擎集成（下一步）：DraftModel 协议 = draft_tokens(hidden, token, k)。
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from cascade.config import ModelConfig
from cascade.model_executor.layers.layernorm import RMSNorm
from cascade.model_executor.layers.linear import Linear
from cascade.model_executor.layers.mla import MLAAttention
from cascade.model_executor.layers.moe import DeepseekMoE, MLP


class DraftModel(Protocol):
    """草案模型接口（引擎集成点）。"""

    def draft_tokens(
        self,
        last_hidden: torch.Tensor,  # [n, H]：主模型末层 hidden（ZOS：上一步的）
        last_token: torch.Tensor,  # [n] int64
        start_positions: torch.Tensor,  # [n] int64：last_token 的绝对位置
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (draft_tokens [n, k] int64, draft_logprobs [n, k, V] fp32,
        p_top1 [n, k] fp32)。draft 采样用独立随机流（MTP_SEED_OFFSET +
        case=1，调用方供种子）。"""
        ...


class NextnDraftModel(nn.Module):
    """DeepSeek-V3 nextn 层（HF 权重命名对齐 vLLM MTP 映射）。

    checkpoint 键（V3）：model.layers.{N}.eh_proj / .shared_head_head /
    .shared_head_norm / 其余 q_a_proj 等与主层同名 —— 加载器映射时把
    第 N 层归到本模块（集群对照真实 checkpoint 校正键名）。
    """

    def __init__(self, cfg: ModelConfig, shared_embed: nn.Embedding, shared_head: nn.Linear) -> None:
        super().__init__()
        assert cfg.has_mtp
        self.cfg = cfg
        layer_idx = cfg.num_layers  # nextn 层序号 = 主层数
        self.eh_proj = Linear(2 * cfg.hidden_size, cfg.hidden_size)
        self.enorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.hnorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.shared_head_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.shared_head_head = shared_head  # 与主模型 lm_head 共享
        self.embed_tokens = shared_embed
        if layer_idx < cfg.num_dense_layers:
            self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        else:
            self.mlp = DeepseekMoE(cfg)

    def forward_step(
        self,
        hidden_prev: torch.Tensor,  # [n, H]
        token: torch.Tensor,  # [n] int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单步 draft：→ (next_hidden [n, H], logits [n, V])。"""
        h = self.eh_proj(torch.cat((hidden_prev, self.embed_tokens(token)), dim=-1))
        h = self.enorm(h)
        attn_out = h  # 层内注意力需 KV/位置 —— 引擎集成时接 cache（TODO）
        x = attn_out
        mlp_out = self.mlp(x)
        next_hidden = self.hnorm(hidden_prev + mlp_out)
        logits = self.shared_head_head(self.shared_head_norm(next_hidden))
        return next_hidden, logits


class MockDraftModel(nn.Module):
    """确定性 mock 草案（V2-Lite 管线测试）。

    draft token = f(seed, 绝对位置) —— 与 tests/cpu/mock_model 的目标
    公式同构；p_top1 取确定值。贪心管线测试中 mock 的接受率无关紧要。
    """

    def __init__(self, vocab_size: int, p_top1: float = 0.6) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.p_top1 = p_top1

    def draft_tokens(
        self,
        last_hidden: torch.Tensor,
        last_token: torch.Tensor,
        start_positions: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from cascade.sampling.rng import derive_uniform

        n = last_token.shape[0]
        device = last_token.device
        tokens = torch.zeros(n, k, dtype=torch.int64, device=device)
        logprobs = torch.zeros(n, k, self.vocab_size, dtype=torch.float32, device=device)
        p_top1 = torch.full((n, k), self.p_top1, dtype=torch.float32, device=device)
        seeds = (last_token.to(torch.int64) * 2654435761) % (1 << 63)  # 请求指纹
        for i in range(n):
            for j in range(k):
                pos = int(start_positions[i]) + j + 1
                u = derive_uniform(int(seeds[i]), pos)
                tokens[i, j] = int(u * self.vocab_size) % self.vocab_size
        # draft 分布：one-hot 在 draft token（贪心语义；p_top1 概率）
        logprobs.fill_(float(torch.log(torch.tensor(1.0 - self.p_top1 / (self.vocab_size - 1)))))
        rows = torch.arange(n, device=device).repeat_interleave(k)
        cols = torch.arange(k, device=device).repeat(n)
        logprobs[rows, cols, tokens.reshape(-1)] = float(torch.log(torch.tensor(self.p_top1)))
        return tokens, logprobs, p_top1


__all__ = ["DraftModel", "NextnDraftModel", "MockDraftModel"]

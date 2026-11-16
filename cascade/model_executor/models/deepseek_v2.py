"""DeepSeek-V2 / V2-Lite model (MLA + MoE, HF 权重命名零映射)。

Reference-quality implementation: 分段注意力（显式/吸收双路径见
`layers/mla.py`），逐专家 MoE —— 集群上与 HF transformers 逐 token 对拍。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cascade.config import ModelConfig
from cascade.model_executor.layers.layernorm import RMSNorm
from cascade.model_executor.layers.mla import MLAAttention
from cascade.model_executor.layers.moe import MLP, DeepseekMoE
from cascade.model_executor.layers.rotary import RotaryEmbedding
from cascade.worker.cache_engine import CacheEngine


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MLAAttention(cfg, layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_idx < cfg.num_dense_layers:
            self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        else:
            self.mlp = DeepseekMoE(cfg)

    def forward(
        self,
        x: torch.Tensor,  # [T, H] 段拼接
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        segment_lens: list[int],
        block_tables: torch.Tensor,
        cache: CacheEngine,
        rope: RotaryEmbedding,
        use_absorbed: bool = False,
        num_splits: int | None = None,
        rope_ensure: bool = True,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x, positions, seq_ids, segment_lens, block_tables, cache, rope,
            use_absorbed, num_splits, rope_ensure,
        )
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class DeepseekV2Model(nn.Module):
    def __init__(self, cfg: ModelConfig, device: str = "cuda") -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(cfg, i) for i in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.rope_theta, device=device)
        self.device = device

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        segment_lens: list[int],
        block_tables: torch.Tensor,
        cache: CacheEngine,
        use_absorbed: bool = False,
        num_splits: int | None = None,
        rope_ensure: bool = True,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(
                x, positions, seq_ids, segment_lens, block_tables, cache, self.rope,
                use_absorbed, num_splits, rope_ensure,
            )
        return self.norm(x)


class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, device: str = "cuda") -> None:
        super().__init__()
        self.cfg = cfg
        self.model = DeepseekV2Model(cfg, device=device)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        segment_lens: list[int],
        block_tables: torch.Tensor,
        cache: CacheEngine,
        use_absorbed: bool = False,
        num_splits: int | None = None,
        rope_ensure: bool = True,
    ) -> torch.Tensor:
        """→ logits [T, vocab]（完整行；采样器自行挑选各段末行/decode 行）。"""
        hidden = self.model(
            input_ids, positions, seq_ids, segment_lens, block_tables, cache,
            use_absorbed, num_splits, rope_ensure,
        )
        return self.lm_head(hidden)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        segment_lens: list[int],
        block_tables: torch.Tensor,
        cache: CacheEngine,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        """→ hidden [T, H]（不含 lm_head；CUDA graph 捕获入口——输出小，
        lm_head 的一次 GEMM 留在图外以省各档位的 logits 显存）。

        rope_ensure 恒 False：图捕获安全（无 .item() 同步）。**调用方必须**
        先 `self.model.rope._ensure(runtime_max_model_len)` 预热缓冲
        （CudaGraphRunner.__init__ 已做；否则捕获后 _ensure 重分配会让
        已捕获的 cos/sin 指针 use-after-free —— 审查 #5）。"""
        return self.model(
            input_ids, positions, seq_ids, segment_lens, block_tables, cache,
            use_absorbed=True, num_splits=num_splits, rope_ensure=False,
        )

    def tie_weights(self) -> None:
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def to(self, *args, **kwargs):
        """nn.Module.to 之外同步 rope 缓冲设备。

        RotaryEmbedding 不是 nn.Module，cos/sin 缓冲不随 model.to 迁移——
        CPU 构造后上卡（真实模型路径）时若不显式迁移，apply 的
        cos[positions] 会因设备不匹配报错（e2e 暴露）。
        """
        super().to(*args, **kwargs)
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            rope = getattr(self.model, "rope", None)
            if rope is not None and hasattr(rope, "to"):
                rope.to(device)
        return self


__all__ = ["DeepseekV2DecoderLayer", "DeepseekV2Model", "DeepseekV2ForCausalLM"]

"""Engine configuration dataclasses.

Split into ModelConfig / CacheConfig / SchedulerConfig so that the CPU-side
scheduler stack (`cascade.core`) depends only on the pieces it needs — this
keeps the scheduler testable on Windows without a GPU or torch.cuda.

`ModelConfig.from_hf_dict` parses a HuggingFace `config.json` for the
DeepSeek-V2 family (V2 / V2-Lite)。deepseek_v3 显式拒绝（关键数学未实现，
见 from_hf_dict；V3 支持落地于 `model_executor/models/deepseek_v3.py`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_MLA_NUM_KV_HEADS = 1  # MLA latent KV: single shared KV stream (MQA-like)


@dataclass
class ModelConfig:
    """Model-side metadata the scheduler, cache, and model layers need."""

    # --- 基础 ---
    model_type: str = "deepseek_v2"
    max_model_len: int = 4096  # max_position_embeddings
    vocab_size: int = 128256  # DeepSeek V3 家族; V2-Lite 以 HF 配置为准
    num_layers: int = 1  # num_hidden_layers
    hidden_size: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    # --- MLA 注意力 ---
    num_attention_heads: int = 16
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    num_kv_heads: int = _MLA_NUM_KV_HEADS

    # --- dense FFN（前 first_k_dense_replace 层）---
    intermediate_size: int = 2048

    # --- MoE ---
    moe_intermediate_size: int = 1408
    first_k_dense_replace: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"  # V2/V2.5/V3 家族; "softmax" 为旧 V2 首版
    topk_method: str = "noaux_tc"  # noaux_tc | greedy | ...
    norm_topk_prob: bool = False
    #: MoE 分组路由（V2-Lite/V2.5/V3 的 n_group/topk_group；0 = 不分组）
    n_group: int = 0
    topk_group: int = 0
    #: RoPE 长度外推配置（V2-Lite: {"type": "yarn", "factor": 40, ...}）
    rope_scaling: dict | None = None

    # --- MTP（V2 无; V3 = 1 层 nextn）---
    num_nextn_predict_layers: int = 0

    # ------------------------------------------------------------------ derived

    @property
    def kv_cache_dim(self) -> int:
        """分页 KV 中每个 token 每层占 `kv_lora_rank + qk_rope_head_dim`（c‖rope）。"""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def qk_head_dim(self) -> int:
        """softmax scale = 1/sqrt(qk_nope + qk_rope)（DeepSeek V2 语义）。"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        """1/sqrt(qk_head_dim)，YaRN 下再乘 mscale²（HF 官方语义）。

        V2-Lite 的 rope_scaling 给出 mscale_all_dim=0.707 → 官方
        `DeepseekV2Attention.__init__` 在 mscale_all_dim 非零时把
        softmax_scale 乘 `yarn_get_mscale(factor, mscale_all_dim) ** 2`
        （≈1.59）。缺这一项会让 attention 分布系统性偏移（e2e 20/20
        分歧的根因）。
        """
        scale = self.qk_head_dim**-0.5
        rs = self.rope_scaling or {}
        mscale_all_dim = rs.get("mscale_all_dim", 0)
        factor = float(rs.get("factor", 1.0))
        if mscale_all_dim and factor > 1:
            m = 0.1 * float(mscale_all_dim) * math.log(factor) + 1.0
            scale *= m * m
        return scale

    @property
    def num_dense_layers(self) -> int:
        return min(self.first_k_dense_replace, self.num_layers)

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers - self.num_dense_layers

    @property
    def has_e_score_correction_bias(self) -> bool:
        return self.topk_method == "noaux_tc"

    @property
    def shared_intermediate_size(self) -> int:
        """共享专家拼接为单个宽 FFN：n_shared × moe_inter（DeepSeek V2 语义）。"""
        return self.n_shared_experts * self.moe_intermediate_size

    @property
    def has_mtp(self) -> bool:
        return self.num_nextn_predict_layers > 0

    # ------------------------------------------------------------------ HF parsing

    @classmethod
    def from_hf_dict(cls, d: dict) -> ModelConfig:
        """从 HF config.json（deepseek_v2 / deepseek_v3 家族）构建。

        只读取字段、给默认值、做硬校验；不做任何架构推断（那是模型层的事）。
        V3 特有字段（quantization_config、rope beta 缩放）留给
        `deepseek_v3.py` 的扩展解析。
        """
        model_type = d.get("model_type", "deepseek_v2")
        if model_type != "deepseek_v2":
            # 审查 #8：V3 的 n_group 组路由 / rope beta 缩放 / nextn 权重映射均未
            # 实现，接受它会造成静默错误的输出（路由与位置编码数学全错）——
            # 显式拒绝而不是半支持。V3 支持落地于 model_executor/models/deepseek_v3.py。
            if model_type == "deepseek_v3":
                raise NotImplementedError(
                    "deepseek_v3 家族暂不支持：n_group 组路由 / rope beta(mscale) / "
                    "nextn 权重映射未实现（当前仅支持 deepseek_v2 = V2/V2-Lite）"
                )
            raise ValueError(f"unsupported model_type {model_type!r} (仅支持 deepseek_v2)")
        cfg = cls(
            model_type=model_type,
            max_model_len=int(d.get("max_position_embeddings", 4096)),
            vocab_size=int(d["vocab_size"]),
            num_layers=int(d["num_hidden_layers"]),
            hidden_size=int(d["hidden_size"]),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=float(d.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
            num_attention_heads=int(d["num_attention_heads"]),
            # q_lora_rank 在真实 V2-Lite config.json 中为 null（无 q LoRA 投影，
            # 走 q_proj 直连路径）——int(None) 会 TypeError
            q_lora_rank=int(d.get("q_lora_rank") or 0),
            kv_lora_rank=int(d["kv_lora_rank"]),
            qk_rope_head_dim=int(d["qk_rope_head_dim"]),
            qk_nope_head_dim=int(d["qk_nope_head_dim"]),
            v_head_dim=int(d["v_head_dim"]),
            num_kv_heads=int(d.get("num_key_value_heads", _MLA_NUM_KV_HEADS)),
            intermediate_size=int(d.get("intermediate_size", 2048)),
            moe_intermediate_size=int(d.get("moe_intermediate_size", 2048)),
            first_k_dense_replace=int(d.get("first_k_dense_replace", 0)),
            n_routed_experts=int(d.get("n_routed_experts", 0)),
            n_shared_experts=int(d.get("n_shared_experts", 0)),
            num_experts_per_tok=int(d.get("num_experts_per_tok", 0)),
            routed_scaling_factor=float(d.get("routed_scaling_factor", 1.0)),
            scoring_func=d.get("scoring_func", "sigmoid"),
            topk_method=d.get("topk_method", "greedy"),
            norm_topk_prob=bool(d.get("norm_topk_prob", False)),
            n_group=int(d.get("n_group") or 0),
            topk_group=int(d.get("topk_group") or 0),
            rope_scaling=d.get("rope_scaling"),
            num_nextn_predict_layers=int(d.get("num_nextn_predict_layers", 0)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert self.num_layers > 0
        assert self.hidden_size > 0
        assert self.vocab_size > 0
        assert self.num_attention_heads > 0
        assert self.qk_nope_head_dim > 0 and self.qk_rope_head_dim > 0 and self.v_head_dim > 0
        assert self.kv_lora_rank > 0
        # 我们的 KV 布局假设 latent 单流（MLA MQA 语义）。
        # 真实 V2-Lite 的 config 把 num_key_value_heads 写成 num_attention_heads
        # （16），但 MLA 的 KV 经 kv_a_proj_with_mqa 压缩为单流 latent——按
        # HF 语义，MLA 下该字段应为 1；此处做归一化而不是拒绝（e2e 真实
        # checkpoint 的 config 即如此）。
        if self.num_kv_heads != 1 and self.kv_lora_rank > 0:
            self.num_kv_heads = 1
        assert self.qk_rope_head_dim % 2 == 0, "rope head dim 必须为偶数"
        assert 0 <= self.first_k_dense_replace <= self.num_layers
        assert self.n_routed_experts >= 0
        assert self.num_experts_per_tok >= 1
        if self.n_routed_experts:
            assert self.num_experts_per_tok <= self.n_routed_experts
        assert self.scoring_func in ("sigmoid", "softmax"), f"不支持的 scoring_func {self.scoring_func}"
        assert self.routed_scaling_factor > 0


@dataclass
class CacheConfig:
    """Paged KV cache geometry.

    `num_gpu_blocks=None` means "derive at runtime from a profile run"
    (Phase 2+); the CPU test harness sets an explicit small value.

    `kv_dtype`:
    - ``"bf16"``：c‖rope 单张量，2 B/元素（默认）
    - ``"fp8"``：latent c 存 FP8(e4m3) + k_rope 存 bf16 + 每 token fp32 scale
      （DeepSeek-V3 生产同构；KV 字节/token/层 1152 → 644 ≈ 1.79× 压缩）
    """

    block_size: int = 64
    num_gpu_blocks: int | None = None
    gpu_memory_utilization: float = 0.9
    kv_dtype: str = "bf16"

    def validate(self) -> None:
        assert self.block_size > 0
        assert self.num_gpu_blocks is None or self.num_gpu_blocks > 0
        assert 0.0 < self.gpu_memory_utilization <= 1.0
        assert self.kv_dtype in ("bf16", "fp8"), f"不支持的 kv_dtype {self.kv_dtype!r}"


@dataclass
class SchedulerConfig:
    """Batching and preemption knobs (DeepInfer-style semantics)."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192  # per-step token budget (chunked prefill cap)
    prefill_sum_seqlen_k: int = 32768  # sum-of-context budget for one prefill batch
    enable_prefix_caching: bool = True
    preemption_recompute: bool = True
    #: MTP 固定验证长度：decode 行的 num_new_tokens = 1 + K
    #: （varlen 调度会按置信率逐请求调整；0 = 关闭 MTP）
    nextn_draft_tokens: int = 0
    #: 置信度调度变长验证（DeepInfer ENABLE_MTP_VARLEN 语义；与批不变
    #: 互斥——验证长度随 batch 组合变化）
    mtp_varlen: bool = False

    def validate(self) -> None:
        assert self.max_num_seqs > 0
        assert self.max_num_batched_tokens > 0
        assert self.prefill_sum_seqlen_k > 0
        assert self.nextn_draft_tokens >= 0
        if self.mtp_varlen:
            assert self.nextn_draft_tokens > 0, "varlen 需要 MTP 开启"
            # 2026-11-18 审计 #6(f)：MtpVarlen.plan 断言 budget ≥ len(rates)
            # ——逐行预算至少 1 token/seq，否则首个 decode 步直接崩溃
            assert self.max_num_batched_tokens >= self.max_num_seqs, (
                f"mtp_varlen 需要 max_num_batched_tokens ({self.max_num_batched_tokens}) "
                f">= max_num_seqs ({self.max_num_seqs})"
            )
            from cascade.common.env import envs

            assert not envs.BATCH_INVARIANT, "varlen 与批不变互斥（DeepInfer 同此）"


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime_max_model_len: int = 0

    def validate(self) -> None:
        self.model.validate()
        self.cache.validate()
        self.scheduler.validate()
        # 交叉校验（审查 #8）：单个 prompt 必须能过 sum-K 预算，否则超长 prompt
        # 会被 sum-K 永久推迟（DeepInfer scheduler.hpp:120-121 的 TORCH_CHECK）
        assert self.scheduler.prefill_sum_seqlen_k >= self.model.max_model_len, (
            f"prefill_sum_seqlen_k ({self.scheduler.prefill_sum_seqlen_k}) 必须 >= "
            f"max_model_len ({self.model.max_model_len})"
        )
        # 位置余量：投机解码需要 max_model_len 之外再容纳 K 个草案位置 + 1 个
        # 回退位（DeepInfer: runtime_max_model_len = max_model_len + 2*nextn + 1；
        # Cascade 的 decode 行预算 = 1+K，K 个草案位置同样要计入 —— 审查 #19
        # 修正：原式漏掉 K，MTP 末步位置可达 max+K-1 越过 runtime）。
        nextn = self.model.num_nextn_predict_layers
        k = self.scheduler.nextn_draft_tokens
        self.runtime_max_model_len = self.model.max_model_len + 2 * nextn + k + 1

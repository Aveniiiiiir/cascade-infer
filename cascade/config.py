"""Engine configuration dataclasses.

Split into ModelConfig / CacheConfig / SchedulerConfig so that the CPU-side
scheduler stack (`cascade.core`) depends only on the pieces it needs — this
keeps the scheduler testable on Windows without a GPU or torch.cuda.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model-side metadata the scheduler and cache need to know about."""

    max_model_len: int = 4096
    vocab_size: int = 128256  # DeepSeek V2 family vocabulary
    num_layers: int = 1  # KV-cache depth (Phase 1 mock models use 1)
    num_kv_heads: int = 1  # MLA: single latent KV (MQA-like)
    kv_lora_rank: int = 512  # MLA latent dim
    qk_rope_head_dim: int = 64  # rope part appended to the latent
    num_nextn_predict_layers: int = 0  # >0 → MTP draft weights exist

    @property
    def kv_cache_dim(self) -> int:
        """Tokens in the paged KV occupy `kv_lora_rank + qk_rope_head_dim` per layer."""
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass
class CacheConfig:
    """Paged KV cache geometry.

    `num_gpu_blocks=None` means "derive at runtime from a profile run"
    (Phase 2+); the CPU test harness sets an explicit small value.
    """

    block_size: int = 64
    num_gpu_blocks: int | None = None
    gpu_memory_utilization: float = 0.9

    def validate(self) -> None:
        assert self.block_size > 0
        assert self.num_gpu_blocks is None or self.num_gpu_blocks > 0
        assert 0.0 < self.gpu_memory_utilization <= 1.0


@dataclass
class SchedulerConfig:
    """Batching and preemption knobs (DeepInfer-style semantics)."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192  # per-step token budget (chunked prefill cap)
    prefill_sum_seqlen_k: int = 32768  # sum-of-context budget for one prefill batch
    enable_prefix_caching: bool = True
    preemption_recompute: bool = True

    def validate(self) -> None:
        assert self.max_num_seqs > 0
        assert self.max_num_batched_tokens > 0
        assert self.prefill_sum_seqlen_k > 0


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime_max_model_len: int = 0

    def validate(self) -> None:
        self.cache.validate()
        self.scheduler.validate()
        # 位置余量：投机解码需要 max_model_len 之外再容纳 K+1 个回退位置
        # (DeepInfer: runtime_max_model_len = max_model_len + 2*nextn + 1)。
        nextn = 0  # Phase 1: 无 MTP；Phase 5 起由 num_nextn_predict_layers 决定
        self.runtime_max_model_len = self.model.max_model_len + 2 * nextn + 1

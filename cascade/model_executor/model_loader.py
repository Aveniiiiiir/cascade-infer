"""Checkpoint loader: safetensors → DeepseekV2ForCausalLM。

模块命名与 HF 一致（零映射加载）**除专家权重外**：MoE 层以堆叠张量持有
专家权重（kernel 按专家索引的要求），HF 的逐专家键
``mlp.experts.E.gate_proj.weight`` 重映射到 ``mlp.w1[E]`` 切片。
V2-Lite bf16 直读；V3 原生 FP8 checkpoint 的 blockwise 反量化在 Phase 6
（quantization_config 存在时报 NotImplementedError）。
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cascade.common.logger import get_logger
from cascade.config import ModelConfig
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

_log = get_logger("model_loader")

_EXPERT_KEY_RE = re.compile(r"^(.*?mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_TARGET = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


@dataclass
class LoadReport:
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    num_tensors: int = 0
    num_bytes: int = 0


def _shard_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text(encoding="utf-8"))
        files = sorted({v for v in idx["weight_map"].values()})
        return [model_dir / f for f in files]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors found under {model_dir}")


def _construct_model(cfg: ModelConfig, dtype: torch.dtype) -> DeepseekV2ForCausalLM:
    """按目标 dtype 直接构造模型骨架（关键内存优化）。

    默认构造是 fp32：V2-Lite 15.7B 参数 × 4B ≈ 63GB 主机内存，且
    `model.to(dtype)` 会在 fp32 骨架之外再分配一份 bf16 副本（瞬时 ~92GB）
    ——62GB 内存机器连加载都过不去。以 bf16 直接构造后骨架 ≈29GB，
    `.to(device, dtype)` 退化为纯搬运（无 dtype 转换副本）。
    """
    prev = torch.get_default_dtype()
    try:
        try:
            torch.set_default_dtype(dtype)
        except (TypeError, RuntimeError):  # pragma: no cover - 老版本 torch
            _log.warning("set_default_dtype(%s) 不可用，按默认 fp32 构造（内存峰值翻倍）", dtype)
        return DeepseekV2ForCausalLM(cfg, device="cpu")
    finally:
        torch.set_default_dtype(prev)


def load_deepseek_v2(
    model_dir: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DeepseekV2ForCausalLM, ModelConfig]:
    """加载 checkpoint（按 dtype 构造 → 逐分片赋值 → 搬运到 device）。

    内存峰值 ≈ 模型骨架（dtype 大小）+ 单个 shard，而非
    骨架 + 全量 state dict —— V2-Lite（~29GB bf16 / 4 个 shard）峰值
    ~37GB，62GB 主机内存可容纳。
    """
    import safetensors.torch  # 局部导入：集群/CI 才有

    model_dir = Path(model_dir)
    hf_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ModelConfig.from_hf_dict(hf_dict)
    fp8_mode = "quantization_config" in hf_dict  # V3 原生 FP8：加载期反量化到 bf16

    model = _construct_model(cfg, dtype)
    files = _shard_files(model_dir)
    report = LoadReport()
    covered: set[str] = set()
    expert_seen: dict[str, set[int]] = {}
    for f in files:
        shard = safetensors.torch.load_file(str(f))
        if fp8_mode:
            # 注：V3 官方 checkpoint 的 *_scale_inv 可能独立分片——按 shard
            # 反量化要求权重与尺度同片（当前 v3 已在 config 层拒绝，
            # 该路径仅 test_fp8_loader 的单片合成 checkpoint 使用）
            shard = _dequant_fp8_state(shard)
        _assign_shard(model, shard, dtype, report, covered, expert_seen)
        del shard
        gc.collect()  # 压住分片副本，峰值只留骨架 + 当前 shard
    _finish_missing(model, report, covered, expert_seen, cfg.tie_word_embeddings)
    if cfg.tie_word_embeddings:
        model.tie_weights()
    model.to(device=device, dtype=dtype, non_blocking=True)
    # RotaryEmbedding 不是 nn.Module——cos/sin 缓冲不随 model.to 迁移，
    # 必须显式迁移（真实模型路径：CPU 构造后上卡）
    rope = getattr(getattr(model, "model", None), "rope", None)
    if rope is not None and hasattr(rope, "to"):
        rope.to(device, dtype)
    _log.info(
        "loaded %s: %d tensors, %.2f GiB, missing=%d unexpected=%d",
        model_dir.name,
        report.num_tensors,
        report.num_bytes / 2**30,
        len(report.missing),
        len(report.unexpected),
    )
    if report.missing or report.unexpected:
        for name in report.missing[:10]:
            _log.warning("missing: %s", name)
        for name in report.unexpected[:10]:
            _log.warning("unexpected: %s", name)
    return model, cfg


def _dequant_fp8_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """V3 原生 FP8 checkpoint 的加载期反量化（Phase 6 先行路径）。

    DeepSeek-V3 格式：权重 float8_e4m3fn + ``<name>_scale_inv``
    [N/128, H/128]（128×128 blockwise）。反量化到 bf16（内存 2×，
    正确性优先；``torch._scaled_mm`` 的在线 fp8 路径后置替换）。
    fp16/bf16 权重原样通过（mixed checkpoint 的 norm 等）。
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith("weight_scale_inv"):
            continue  # 尺度已被消费（随其权重键）
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale_key = f"{key}_scale_inv"
            scale = state.get(scale_key)
            if scale is None:
                _log.warning("FP8 权重 %s 缺少 %s，按 scale=1 反量化", key, scale_key)
                out[key] = tensor.to(torch.bfloat16)
                continue
            out[key] = _dequant_blockwise(tensor, scale)
        else:
            out[key] = tensor
    return out


def _dequant_blockwise(w: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """[N, H] fp8 × [ceil(N/128), ceil(H/128)] 尺度 → bf16。

    尺度经 repeat_interleave 扩展到 block 网格后**裁剪**到 [N, H]——
    兼容非 128 对齐的权重（如 rank=32 的 q_a_proj），不再硬断言对齐。
    """
    N, H = w.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)  # [ceil*Nblk, ceil*Hblk]
    s = s[:N, :H]  # 裁剪到实际权重形状（非 128 对齐时去尾）
    return (w.to(torch.float32) * s).to(torch.bfloat16)


def _assign_shard(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    report: LoadReport,
    covered: set[str],
    expert_seen: dict[str, set[int]],
) -> None:
    """把一个 shard 的键值赋进模型参数（流式加载的单片入口）。

    covered / expert_seen 跨片累积；missing 由 _finish_missing 收尾计算。
    """
    own = model.state_dict()
    for key, tensor in state.items():
        report.num_tensors += 1
        report.num_bytes += tensor.numel() * tensor.element_size()
        m = _EXPERT_KEY_RE.match(key)
        if m:
            # 逐专家键 → 堆叠切片
            prefix, expert_idx, which = m.group(1), int(m.group(2)), m.group(3)
            target_name = f"{prefix}.{_EXPERT_TARGET[which]}"
            if target_name not in own:
                report.unexpected.append(key)
                continue
            target = own[target_name]
            if not (0 <= expert_idx < target.shape[0]):
                report.unexpected.append(key)
                continue
            target[expert_idx].copy_(tensor.to(dtype))
            expert_seen.setdefault(target_name, set()).add(expert_idx)
            continue
        if key not in own:
            report.unexpected.append(key)
            continue
        target = own[key]
        # e_score_correction_bias 保持 checkpoint 原精度（HF 语义 fp32）；其余 cast
        if "e_score_correction_bias" in key:
            target.copy_(tensor.to(target.dtype))
        else:
            target.copy_(tensor.to(dtype))
        covered.add(key)


def _finish_missing(
    model: torch.nn.Module,
    report: LoadReport,
    covered: set[str],
    expert_seen: dict[str, set[int]],
    tie_word_embeddings: bool,
) -> None:
    """审查 #12：missing 报告保真——
    - lm_head.weight 仅在 tie_word_embeddings 时从 missing 排除（此前无条件
      排除：非 tie checkpoint 缺 lm_head → 随机初始化 → 静默垃圾输出）；
    - 专家堆叠参数按 (name, expert_idx) 粒度记录（此前任一专家键加载就把
      整个 w1/w3/w2 标记 covered：部分专家缺失被掩盖 → 零权重专家）。
    """
    own = model.state_dict()
    report.missing = [
        k
        for k in own
        if k not in covered
        and k not in expert_seen
        and not (tie_word_embeddings and k == "lm_head.weight")
    ]
    for name, seen in expert_seen.items():
        num_experts = own[name].shape[0]
        report.missing.extend(f"{name}[{e}]" for e in range(num_experts) if e not in seen)


def _assign(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    tie_word_embeddings: bool = False,
) -> LoadReport:
    """整 state dict 的便利入口（测试/小 checkpoint 用；生产走流式）。"""
    report = LoadReport()
    covered: set[str] = set()
    expert_seen: dict[str, set[int]] = {}
    _assign_shard(model, state, dtype, report, covered, expert_seen)
    _finish_missing(model, report, covered, expert_seen, tie_word_embeddings)
    return report


__all__ = ["load_deepseek_v2", "LoadReport"]

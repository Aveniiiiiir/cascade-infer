"""Checkpoint loader: safetensors → DeepseekV2ForCausalLM。

模块命名与 HF 一致（零映射加载）**除专家权重外**：MoE 层以堆叠张量持有
专家权重（kernel 按专家索引的要求），HF 的逐专家键
``mlp.experts.E.gate_proj.weight`` 重映射到 ``mlp.w1[E]`` 切片。
V2-Lite bf16 直读；V3 原生 FP8 checkpoint 的 blockwise 反量化在 Phase 6
（quantization_config 存在时报 NotImplementedError）。
"""

from __future__ import annotations

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


def load_deepseek_v2(
    model_dir: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DeepseekV2ForCausalLM, ModelConfig]:
    """加载 checkpoint（CPU 组装 → cast → 搬运到 device）。"""
    import safetensors.torch  # 局部导入：集群/CI 才有

    model_dir = Path(model_dir)
    hf_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ModelConfig.from_hf_dict(hf_dict)
    fp8_mode = "quantization_config" in hf_dict  # V3 原生 FP8：加载期反量化到 bf16

    model = DeepseekV2ForCausalLM(cfg, device="cpu")
    files = _shard_files(model_dir)
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(safetensors.torch.load_file(str(f)))
    if fp8_mode:
        state = _dequant_fp8_state(state)
    report = _assign(model, state, dtype, tie_word_embeddings=cfg.tie_word_embeddings)
    if cfg.tie_word_embeddings:
        model.tie_weights()
    model.to(device=device, dtype=dtype, non_blocking=True)
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


def _assign(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    tie_word_embeddings: bool = False,
) -> LoadReport:
    """审查 #12：missing 报告保真——
    - lm_head.weight 仅在 tie_word_embeddings 时从 missing 排除（此前无条件
      排除：非 tie checkpoint 缺 lm_head → 随机初始化 → 静默垃圾输出）；
    - 专家堆叠参数按 (name, expert_idx) 粒度记录（此前任一专家键加载就把
      整个 w1/w3/w2 标记 covered：部分专家缺失被掩盖 → 零权重专家）。
    """
    report = LoadReport()
    own = model.state_dict()
    covered: set[str] = set()  # 已赋值的普通参数名
    expert_seen: dict[str, set[int]] = {}  # 堆叠参数名 → 已加载的专家切片
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
    return report


__all__ = ["load_deepseek_v2", "LoadReport"]

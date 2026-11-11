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
    if "quantization_config" in hf_dict:
        raise NotImplementedError(
            "FP8 checkpoint（DeepSeek-V3）反量加载在 Phase 6；当前请用 V2/V2-Lite bf16 checkpoint"
        )

    model = DeepseekV2ForCausalLM(cfg, device="cpu")
    files = _shard_files(model_dir)
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(safetensors.torch.load_file(str(f)))
    report = _assign(model, state, dtype)
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


def _assign(model: torch.nn.Module, state: dict[str, torch.Tensor], dtype: torch.dtype) -> LoadReport:
    report = LoadReport()
    own = model.state_dict()
    covered: set[str] = set()  # 已赋值的模型参数名（堆叠参数按名标记）
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
            covered.add(target_name)
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
    report.missing = [k for k in own if k not in covered and "lm_head.weight" not in k]
    return report


__all__ = ["load_deepseek_v2", "LoadReport"]

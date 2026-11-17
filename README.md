# Cascade

生产级 LLM 推理引擎。以 DeepSeek 的 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 为设计蓝本，用 **PyTorch + Triton 自研算子**路线实现其全部核心能力，运行于通用 CUDA GPU（不绑定 SM100 / DeepGEMM / FlashMLA）。

## 特性（对标 DeepInfer）

- 连续批处理（混合 varlen 批：decode + chunked prefill 同步推进）
- 分页 KV cache + 前缀缓存（block 哈希链 + LRU + pin/unpin + COW）
- ZOS 零开销异步调度（三段式 `Forward(N) → PostProcess(N−1) → Sample(N)`，乐观推进 + 批间 remap）
- MTP 投机解码 + 置信度调度变长验证（采样式拒绝采样，累乘接受率 + 性能曲线定预算）
- 批不变推理 + 确定性采样（RNG 种子 = f(seq_seed, token 绝对位置, 用途流)）
- 结构化输出（xgrammar：regex / JSON schema / EBNF，异步编译 + 失败隔离）
- CUDA Graph（按 total_q 分档、全档共享 memory pool）
- MoE 专家并行 EP + EPLB（NCCL 路线）
- FP8 权重与 KV cache
- OpenAI 兼容 HTTP 服务 + 离线 LLM API

## 支持模型

- DeepSeek-V2 / V2-Lite（MLA + MoE）—— 主测试模型
- DeepSeek-V3 / R1（MLA + MoE + nextn MTP）

## 快速开始

```bash
pip install -e ".[dev,so]"

# 离线推理（需要 GPU 主机）
python -m cascade.entrypoints.llm --model deepseek-ai/DeepSeek-V2-Lite

# CPU 单元测试（Windows 开发机可全跑）
pytest tests/cpu
```

## 开发流

本仓库在 Windows 上开发纯 Python 逻辑（调度器、参考实现、CPU 测试），GPU 相关验收在 Linux 集群执行：

```bash
pytest tests/cpu            # 本机守卫，必须全绿
scripts/gpu_smoke.sh        # 集群冒烟
```

## 验收状态（2026-11-17）

集群 e2e（RTX 4080 SUPER 32GB，DeepSeek-V2-Lite 对拍 HF 官方 modeling）：

| 判据 | 结果 |
|---|---|
| teacher-forcing explicit | **632/640 = 98.75%** 逐位置 argmax 一致 |
| teacher-forcing absorbed（MLA absorbed decode） | **633/640 = 98.91%** |
| free-running 混合批 | 首分歧均位于 HF margin ≤ 0.125 的 tie 级位置 |

- 分歧全部为 bf16 tie 级数值噪声，非系统性错误；验收以 teacher-forcing 一致率
  与 margin 检查为准（bf16 下 free-running 不保证 token-exact）。
- 非 e2e 全量测试（3080 Ti / 4080）：`tests/{cpu,model,triton_ops,engine}` 全绿。
- 性能基线（3080 Ti，生产形状）：decode 381 GB/s（8.1e4 tok/s）、
  prefill 2.25e6 tok/s、fused_moe 18.5 TFLOPS。

## 设计文档

- 架构与分阶段计划：见 `docs/`
- 问题与修复报告：`docs/cascade-infer-问题与修复报告.md`（50+ 项）
- 集群验收进度：`docs/集群验收进度-2026-11-17.md`
- 上游参考：DeepInfer 源码（只读）

## License

Apache-2.0

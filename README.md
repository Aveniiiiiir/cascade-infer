# Cascade

生产级 LLM 推理引擎。以 DeepSeek 的 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 为设计蓝本，用 **PyTorch + Triton 自研算子**路线实现其全部核心能力，运行于通用 CUDA GPU（不绑定 SM100 / DeepGEMM / FlashMLA）。

## 特性（对标 DeepInfer）

- 连续批处理（混合 varlen 批：decode + chunked prefill 同步推进）
- 分页 KV cache + 前缀缓存（block 哈希链 + LRU + pin/unpin + COW）
- 同步序引擎步（schedule → prepare → forward → sample → commit；
  ZOS 零开销异步三段式为后置增强——需 GPU 侧 token 喂数落地）
- MTP 投机解码 + 置信度调度变长验证（采样式拒绝采样，累乘接受率 + 性能曲线定预算）
- 确定性采样（RNG 种子 = f(seq_seed, token 绝对位置, 用途流)）+ 同批形状
  重放 bit 级一致；**跨批形状 bit 级一致为 DeepInfer 级目标**（当前前向
  的 MoE 分桶/注意力分块对批形状敏感，见 test_batch_invariance 的 xfail）
- 结构化输出（xgrammar：regex / JSON schema / EBNF，异步编译 + 失败隔离）
- CUDA Graph（按 batch 档位分档、全档共享 memory pool；真实 V2-Lite 已验证）
- MoE 专家并行 EP + EPLB（代码就绪，**多卡联调为后置项**；单卡路径已验收）
- FP8 权重加载期反量化（在线 fp8 GEMM 与 FP8 KV cache 为后置项）
- OpenAI 兼容 HTTP 服务 + 离线 LLM API（真实模型 serving 已验收）

## 支持模型

- **DeepSeek-V2 / V2-Lite**（MLA + MoE）—— 主测试模型，已全链路验收
- DeepSeek-V3 / R1 —— **暂不支持**（`config.py` 显式拒绝：n_group 组路由 /
  rope beta 缩放 / nextn 权重映射未实现；支持落地于 roadmap）

## 快速开始

```bash
pip install -e ".[dev,so]"

# 离线推理（需要 GPU 主机 + 本地已下载的 HF 格式模型目录）
python -m cascade.entrypoints.llm --model /path/to/DeepSeek-V2-Lite

# CPU 单元测试（Windows 开发机可全跑）
pytest tests/cpu
```

## 开发流

本仓库在 Windows 上开发纯 Python 逻辑（调度器、参考实现、CPU 测试），GPU 相关验收在 Linux 集群执行：

```bash
pytest tests/cpu            # 本机守卫，必须全绿
scripts/gpu_smoke.sh        # 集群冒烟
```

## 验收状态（2026-11-23）

集群（RTX 4080 SUPER 32GB，DeepSeek-V2-Lite 对拍 HF 官方 modeling）：

| 判据 | 结果 |
|---|---|
| teacher-forcing explicit | **190/192 = 98.96%** 逐位置 argmax 一致 |
| teacher-forcing absorbed（MLA absorbed decode） | **190/192 = 98.96%** |
| free-running 混合批 | 首分歧均位于 HF margin ≤ 0.125 的 tie 级位置 |
| 分层测试 `tests/{cpu,model,triton_ops,engine}` | 全绿（含 7 项回归 + 2 xfail 批不变跨形状） |
| 真实模型 CUDA Graph | **11 档 [1..64] 生产档位，graph ≡ eager bit 级一致** |
| 真实模型 HTTP serving | 通过（auth/SSE/并发/断连恢复/metrics） |

### 性能（4080 SUPER 32GB，V2-Lite bf16）

单 seq（引擎级）：

| 指标 | eager | CUDA graph | 加速比 |
|---|---|---|---|
| TPOT | 64.9ms | **20.3ms** | **3.2x** |
| TTFT | 0.08s | 0.11s | — |

HTTP 服务（SSE 流式，CUDA graph，`scripts/load_test.py`）：

| 并发 | QPS | tokens/s | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 8 | 7.7 | 245.9 | 160ms | 21.7ms |
| 16 | 16.6 | 530.0 | 188ms | 22.5ms |
| 32 | **23.8** | **762.9** | 226ms | 28.3ms |

- `acceptance_full.sh` 端到端通过（`ACCEPTANCE FULL OK`）
- 32 并发 / 762.9 tok/s 为 32GB 卡上限；更高并发建议 48GB+

- 分歧全部为 bf16 tie 级数值噪声，非系统性错误；验收以 teacher-forcing 一致率
  与 margin 检查为准（bf16 下 free-running 不保证 token-exact）。
- kernel 基线（3080 Ti，生产形状）：decode 381 GB/s（8.1e4 tok/s）、
  prefill 2.25e6 tok/s、fused_moe 18.5 TFLOPS。
- 跨批形状 bit 级一致为 DeepInfer 级目标（MoE 分桶/注意力分块形状敏感，
  见 test_batch_invariance 的 xfail）；确定性采样与同形状重放已确认。

## 设计文档

- 架构与分阶段计划：见 `docs/`
- 问题与修复报告：`docs/cascade-infer-问题与修复报告.md`（50+ 项）
- 集群验收进度：`docs/集群验收进度-2026-11-17.md`
- 上游参考：DeepInfer 源码（只读）

## License

Apache-2.0

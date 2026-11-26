# CLAUDE.md — Cascade 开发指引（Agent 上下文）

## 项目

生产级 LLM 推理引擎（PyTorch + Triton 自研算子，DeepSeek-V2-Lite 为主测试模型）。
对标 DeepSeek 的 DeepInfer 框架；V4.1 专属机制（mHC/YOCO/DSA）不实现。

## 开发流（勿改变）

- **Windows 本机**：只跑 `pytest tests/cpu`（不依赖 torch，torch 相关自动 skip）。
  本机网络差，**不要安装 torch**；涉 torch 的验证一律去集群。
- **Linux GPU 集群**：`tests/{model,triton_ops,engine}` + `tests/e2e` +
  `scripts/acceptance_full.sh`。e2e 需 `CASCADE_E2E_MODEL=<本地模型目录>`。
- 正确性门槛：本地 CPU 全绿 → 集群四套全绿 → e2e（teacher-forcing 一致率
  explicit ≥98.5%、absorbed ≥98%，分歧仅允许 HF margin<0.25 的 tie 级）。

## 关键设计决定（勿推翻，推翻前先读对应 docstring/审计报告）

1. **引擎同步序**（schedule → prepare → forward → sample → commit）：
   DeepInfer 的 ZOS 一步滞后依赖 GPU 侧 token 喂入，Cascade CPU 组批下
   数学不可行。GPU 侧喂数落地前不得恢复一步滞后。
2. **CUDA graph 只捕获 forward_hidden**（lm_head 留图外）；padding 行恒指
   scratch 块；`graph_pad_to` 上界 = T·K + E·bm。
3. **BlockManager**：block id 1..num_blocks，行 0 保留；`capacity =
   num_blocks − reserved`（graph scratch 计入 reserved）。容量守卫一律用
   `capacity`，不得直读 `num_blocks`。
4. **MLA 双路径**：explicit（HF 对拍 oracle）/ absorbed（生产，`CASCADE_ABSORBED`
   开关）。KV 布局 `[layers, num_blocks+1, block_size, kv_lora+rope_dim]`。
5. **MoE 门控走 bf16 矩乘 + fp32 softmax**——4080/torch2.7 实测一致率高于
   官方 fp32 路径（638 vs 631/640）；换环境需重测（见 moe.py docstring）。
6. **确定性采样**：RNG = f(seq_seed, 绝对位置, 用途流)；`derive_num_blocks`
   按保留行 (n+1) 计。
7. **投机解码双源**：MTP（需要 nextn 权重）与 n-gram（无 draft 模型，
   `CASCADE_NGRAM_DRAFT=1`）。n-gram 草案是 one-hot 稀疏分布，走
   `mtp_verify(draft_token_probs=...)`（不分配 [K, V]）；CUDA graph 的
   多 token 验证行要求整批 v=1+K 一致，`uniform_v` 路径把段元数据全部
   张量化（positions[::v]+v 推导），varlen/动态长度自动退化 eager。
8. **FP8 KV**：`CASCADE_KV_DTYPE=fp8` 或 `--kv-cache-dtype fp8`。布局 =
   c(float8_e4m3) + rope(bf16) + 每 128 维一组 fp32 scale；量化在 torch 侧
   完成、kernel 只搬字节（规避 sm89 上 Triton fp8 cast 的版本差异）；
   显式路径 gather 必须用 int8 视图（CPU 的 fp8 高级索引未实现）。

## 陷阱（历史上真实炸过）

- `__slots__` 类新增属性必须同步声明 `__slots__`（tests/cpu/test_slots_consistency.py
  有 AST 检查，但别依赖它兜底）。
- 图捕获区间禁止 CPU-GPU 同步（`.item()`/`.max()`）；rope 缓冲必须引擎构造期
  预热，热路径 `rope_ensure=False`。
- `EngineConfig.validate()` 在 LLMEngine 无条件执行；`llm.py` 的
  `prefill_sum_seqlen_k` 必须对齐 max_model_len。
- Triton kernel 内部参数不加 Python 类型注解（JIT 按值特化，pyproject 已豁免 ANN）。
- 测试中 monkeypatch `compiler.HAS_XGRAMMAR=False` 可强制 stub 编译路径——
  装有 xgrammar 的集群也要真正执行 SO 测试（历史覆盖陷阱）。

## 结构速查

```
cascade/core/       调度器、块管理、前缀缓存（纯 Python，Windows 可测）
cascade/engine/     LLMEngine（同步序）+ EngineStepError（回执携带）
cascade/worker/     model_runner / cache_engine / cuda_graph_runner / kv_layout
cascade/model_executor/  layers(mla/rope/moe) + triton_ops + model_loader
cascade/entrypoints/     llm.py（离线）/ http/server.py（OpenAI 兼容）
tests/cpu|model|triton_ops|engine|e2e   验收分层
scripts/            acceptance_full.sh + debug_*（一次性诊断工具）
docs/               设计文档 + 审计/验收报告
```

## 提交规范

- 默认分支 master + code-review-fixes 同步推进；commit 结尾
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 未明确要求不提交/不推送。集群验收后先跑全量测试再提报告。

## 报告位置

- 本地：`D:\cascade-reports\`（审计/验收全套历史）
- 仓库：`docs/`（入库版）

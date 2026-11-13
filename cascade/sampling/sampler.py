"""GPU 采样编排：温度/top-p/top-k 变换 + 确定性 RNG 采样 + SO 掩码 + golden。

对应 DeepInfer `layers/sampler.py` 的编排职责（其变换+采样融合在
tilelang fused_sample kernel 中；Cascade 的 Triton fused 版在 Phase 5 与
MTP 拒绝采样一起落地，当前 torch 变换 + det_random_sampling 组合）。

顺序契约（与 DeepInfer 一致）：
1. SO 位掩码 → logits 原地 -inf（约束在变换前生效）
2. 温度缩放 → top-k 截断 → top-p 截断 → softmax → probs
3. det_random_sampling（种子 = f(seq_seed, 绝对位置, case=0)）
4. golden teacher-forcing 覆盖（测试/回放）

贪心无需特判：temperature < 1e-5 时 SamplingParams 已退化为
(eps, top_p=1, top_k=-1)（DeepInfer sampling_params.py:76-79），
softmax 在 eps 温度下欠溢为 one-hot ⇒ 逆 CDF 首穿越列 = argmax（tie 取
最小 index，与 argmax 一致）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cascade.model_executor.triton_ops.apply_so_bitmask import apply_token_bitmask
from cascade.model_executor.triton_ops.det_random_sampling import det_random_sampling
from cascade.sampling_params import SamplingScalars


@dataclass(slots=True)
class SamplingRows:
    """展平的待采样行（每行 = 一个 decode 行或 prefill 收尾行）。"""

    temperatures: torch.Tensor  # [R] fp32
    top_ps: torch.Tensor  # [R] fp32
    top_ks: torch.Tensor  # [R] int32（-1 = 不截断）
    seq_seeds: torch.Tensor  # [R] int64（uint64 位模式）
    positions: torch.Tensor  # [R] int64：新 token 的绝对位置
    goldens: torch.Tensor  # [R] int32（-1 = 无 teacher forcing）
    request_ids: list[str]


class Sampler:
    __slots__ = ()

    def sample(
        self,
        logits: torch.Tensor,  # [R, vocab] fp32（原地修改：SO 掩码）
        rows: SamplingRows,
        so_bitmask: torch.Tensor | None = None,  # [n_rows, ceil(vocab/32)] int32
        so_row_map: torch.Tensor | None = None,  # [R] int32
    ) -> list[int]:
        """同步采样 → CPU token 列表（测试/同步路径）。"""
        return self.sample_tensors(logits, rows, so_bitmask, so_row_map).tolist()

    def sample_tensors(
        self,
        logits: torch.Tensor,
        rows: SamplingRows,
        so_bitmask: torch.Tensor | None = None,
        so_row_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GPU 采样 → [R] int32（ZOS：launch 后异步 D2H，不碰 CPU）。"""
        assert logits.ndim == 2 and logits.is_contiguous() and logits.dtype == torch.float32
        R, vocab = logits.shape
        assert rows.seq_seeds.shape == (R,), f"{rows.seq_seeds.shape} vs ({R},)"
        # 1. SO 掩码（Phase 5 的 matcher 在此供给位掩码）
        if so_bitmask is not None:
            apply_token_bitmask(logits, so_bitmask, so_row_map)
        # 2. 变换 → probs
        probs = self._transform(logits, rows)
        # 3. 确定性采样
        out = det_random_sampling(probs, rows.seq_seeds, rows.positions, case=0, seed_offset=0)
        # 4. golden teacher-forcing（GPU 侧 where，免 CPU 往返）
        if rows.goldens.numel() and bool((rows.goldens >= 0).any()):
            out = torch.where(rows.goldens >= 0, rows.goldens, out)
        return out

    # ------------------------------------------------------------------ transforms

    def _transform(self, logits: torch.Tensor, rows: SamplingRows) -> torch.Tensor:
        """温度 → top-k → top-p → softmax（torch 变换；fused kernel Phase 5）。

        top-k 语义：保留 ≥ 该行第 k 大值的所有列（tie 保留，确定性）。
        top-p 语义：按概率降序的最小前缀使累积 ≥ p；保留判据
        `cum - prob < p`（首个 token 恒保留），p == 1 时全保留。
        """
        R, vocab = logits.shape
        x = logits / rows.temperatures[:, None]
        # ---- top-k（每行独立 k；k ≤ 0 或 ≥ vocab 不截断）
        ks = rows.top_ks
        if bool((ks > 0).any()):
            sorted_x = x.sort(dim=-1, descending=True).values
            # 审查 #9：k ≥ vocab 等价不截断（kth=行最小值 ⇒ 无列被掩掉），
            # 但 gather 索引必须落在 [0, vocab-1]——否则越界崩溃
            ks_eff = torch.where((ks > 0) & (ks < vocab), ks, torch.full_like(ks, vocab))
            k_idx = torch.clamp(ks_eff, min=1).to(torch.int64) - 1
            kth = sorted_x.gather(1, k_idx[:, None]).squeeze(1)  # 每行第 k 大
            kth = torch.where((ks > 0) & (ks < vocab), kth, torch.full_like(kth, float("-inf")))
            x = x.masked_fill(x < kth[:, None], float("-inf"))
        probs = torch.softmax(x, dim=-1)
        # ---- top-p（每行独立 p）
        ps = rows.top_ps
        if bool((ps < 1.0).any()):
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            keep_sorted = (cum - sorted_probs) < ps[:, None]
            keep_sorted = keep_sorted | (ps >= 1.0)[:, None]  # p==1 全保留
            keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_idx, keep_sorted)
            probs = probs * keep
        return probs


def build_sampling_rows(
    per_row: list[tuple[str, SamplingScalars, int, int]],  # (rid, snapshot, 新 token 绝对位置, golden)
    device: str,
) -> SamplingRows:
    """构建展平采样行。golden = snapshot.golden_at(该请求当前 output 长度)，
    由调用方（model_runner）计算。"""
    if not per_row:
        return SamplingRows(
            temperatures=torch.zeros(0, dtype=torch.float32, device=device),
            top_ps=torch.zeros(0, dtype=torch.float32, device=device),
            top_ks=torch.zeros(0, dtype=torch.int32, device=device),
            seq_seeds=torch.zeros(0, dtype=torch.int64, device=device),
            positions=torch.zeros(0, dtype=torch.int64, device=device),
            goldens=torch.zeros(0, dtype=torch.int32, device=device),
            request_ids=[],
        )
    return SamplingRows(
        temperatures=torch.tensor([s.temperature for _, s, _, _ in per_row], dtype=torch.float32, device=device),
        top_ps=torch.tensor([s.top_p for _, s, _, _ in per_row], dtype=torch.float32, device=device),
        top_ks=torch.tensor([s.top_k for _, s, _, _ in per_row], dtype=torch.int32, device=device),
        seq_seeds=torch.tensor([s.seq_seed for _, s, _, _ in per_row], dtype=torch.int64, device=device),
        positions=torch.tensor([p for _, _, p, _ in per_row], dtype=torch.int64, device=device),
        goldens=torch.tensor([g for _, _, _, g in per_row], dtype=torch.int32, device=device),
        request_ids=[rid for rid, _, _, _ in per_row],
    )


__all__ = ["Sampler", "SamplingRows", "build_sampling_rows"]

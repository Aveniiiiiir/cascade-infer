"""离线推理入口：LLM API + CLI（集群 GPU 上运行）。

用法:
    python -m cascade.entrypoints.llm --model /path/to/DeepSeek-V2-Lite \
        --prompt "你好" --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class LLM:
    """离线推理门面（tokenizer 惰性导入 transformers）。"""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        max_model_len: int | None = None,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        enable_prefix_caching: bool = True,
        num_blocks: int | None = None,
    ) -> None:
        import torch

        from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
        from cascade.engine.llm_engine import LLMEngine

        self._model_dir = model_dir
        hf = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        model_cfg = ModelConfig.from_hf_dict(hf)
        if max_model_len is not None:
            model_cfg.max_model_len = max_model_len
        # eos 回退链（审查 #13）：config.json → generation_config.json →
        # DeepSeek V2 家族默认 3。旧默认 100001 在 config 缺键时永不命中
        # EOS，全部生成拖到 length cap。
        gen: dict = {}
        gen_path = Path(model_dir) / "generation_config.json"
        if gen_path.exists():
            gen = json.loads(gen_path.read_text(encoding="utf-8"))
        self.eos_token_id = int(hf.get("eos_token_id", gen.get("eos_token_id", 3)))
        cfg = EngineConfig(
            model=model_cfg,
            cache=CacheConfig(),
            scheduler=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                # 审查 #8 交叉校验要求 sum-K 预算 ≥ max_model_len；真实
                # V2-Lite 的 max_position_embeddings=163840 > 默认 32768，
                # 未对齐时 LLMEngine.validate()（2026-11-18 起无条件执行）
                # 会直接拒载。单个 prompt 必须能过预算是设计语义。
                prefill_sum_seqlen_k=max(SchedulerConfig.prefill_sum_seqlen_k, model_cfg.max_model_len),
                enable_prefix_caching=enable_prefix_caching,
            ),
        )
        self._engine = LLMEngine(cfg, model_dir, device=device, dtype=torch.bfloat16, num_blocks=num_blocks)
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # 安全注记（审查 #18）：trust_remote_code 会执行模型目录内的
            # 自定义 Python——仅当 model_dir 可信时使用（本机路径/官方仓库）。
            # 不可信来源的模型必须先审计其 tokenizer 代码。
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir, trust_remote_code=True)
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> list[str]:
        from cascade.sampling_params import SamplingParams

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seq_seed=seed if seed is not None else 0,
            stop_token_ids=(self.eos_token_id,),
        )
        tok = self.tokenizer
        ids = [tok.encode(p) for p in prompts]
        results = self._engine.generate(ids, sp)
        # add_request 按序生成 req-N；generate 返回 dict → 按入参顺序取出
        ordered = [results[f"req-{i}"] for i in range(len(ids))]
        return [tok.decode(toks) for toks in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade offline inference")
    parser.add_argument("--model", required=True, help="模型目录（HF 格式 safetensors）")
    parser.add_argument("--prompt", action="append", help="可重复；不传则交互模式")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=not args.no_prefix_cache,
    )
    if args.prompt:
        outs = llm.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        for p, o in zip(args.prompt, outs):
            print(f"---\n{p}\n{o}")
        return
    # 交互模式：逐条生成
    while True:
        try:
            line = input("prompt> ")
        except EOFError:
            break
        if not line.strip():
            continue
        print(llm.generate([line], max_tokens=args.max_tokens, temperature=args.temperature)[0])


if __name__ == "__main__":
    main()

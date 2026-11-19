"""打印 explicit 路径 gather_range_reference 的实际参数（定位越界）。"""

from __future__ import annotations

import os

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPT = "The capital of France is"


def main() -> None:
    from cascade.entrypoints.llm import LLM
    from cascade.worker.cache_engine import CacheEngine

    orig = CacheEngine.gather_range_reference
    calls = []

    def dbg(self, layer_idx, seq_row, num_tokens, block_tables):
        calls.append((layer_idx, seq_row, num_tokens, tuple(block_tables.shape), self.num_blocks))
        if len(calls) <= 40 or len(calls) % 200 == 0:
            print(f"gather L{layer_idx} row={seq_row} ntok={num_tokens} "
                  f"bt={tuple(block_tables.shape)} pool={self.num_blocks}", flush=True)
        return orig(self, layer_idx, seq_row, num_tokens, block_tables)

    CacheEngine.gather_range_reference = dbg
    try:
        llm = LLM(MODEL_DIR, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096)
        out = llm.generate([PROMPT], max_tokens=4, temperature=0.0)
        print("generate OK:", repr(out[0]))
    except Exception as e:
        print("FAIL:", type(e).__name__, str(e)[:200])
        print("last calls:", calls[-6:])
        raise


if __name__ == "__main__":
    main()

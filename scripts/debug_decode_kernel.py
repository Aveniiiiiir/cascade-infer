"""单层 MLA decode：explicit vs absorbed(ref) vs decode-kernel 三方对拍。"""

from __future__ import annotations

import copy
import os

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")


def main() -> None:
    from transformers import AutoTokenizer

    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    attn = model.model.layers[0].self_attn
    rope = model.model.rope
    embed = model.model.embed_tokens

    ids = tok.encode("The capital of France is")
    P = len(ids)
    a0 = 8913

    cache = CacheEngine(
        cfg, CacheConfig(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16
    )
    bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
    bt[0, 0] = 1
    zeros = torch.zeros(1, dtype=torch.long, device="cuda")

    with torch.no_grad():
        # prefill 写入 0..P-1 的 KV（层 0）
        x_p = embed(torch.tensor(ids, device="cuda"))
        qn_p, qr_p = attn.project_q(x_p)
        c_p, k_p = attn.project_kv(x_p)
        qr_p, kr_p = rope.apply(qr_p, k_p, torch.arange(P, device="cuda"))
        cache.store_latent(
            0,
            torch.cat((c_p, kr_p), dim=-1),
            torch.zeros(P, dtype=torch.long, device="cuda"),
            torch.arange(P, device="cuda"),
            bt,
        )

        # decode token 的投影
        pos = torch.tensor([P], device="cuda")
        x_d = embed(torch.tensor([a0], device="cuda"))
        qn_d, qr_d = attn.project_q(x_d)
        c_d, k_d = attn.project_kv(x_d)
        qr_d, kr_d = rope.apply(qr_d, k_d, pos)

        # 1) explicit vs absorbed（整层，各自 cache 副本）
        cache_exp = copy.deepcopy(cache)
        cache_abs = copy.deepcopy(cache)
        o_exp = attn(x_d, pos, zeros, [1], bt, cache_exp, rope, use_absorbed=False)
        o_abs = attn(x_d, pos, zeros, [1], bt, cache_abs, rope, use_absorbed=True)
    print(
        "layer0 decode: explicit vs absorbed max|d| =",
        float((o_exp - o_abs).abs().max()),
    )

    # 2) 直接对拍 kernel vs ref（吸收形态输入；cache 已被上面的 forward 写入 pos P）
    from cascade.model_executor.triton_ops.paged_mla_decode_attn import (
        paged_mla_decode_attn,
        paged_mla_decode_attn_ref,
    )

    with torch.no_grad():
        q_abs = torch.einsum("thd,hdk->thk", qn_d.float(), attn.u_k.float())
        q_full = torch.cat((q_abs, qr_d.float()), dim=-1).to(cache.kv_cache.dtype)
        kv = cache.layer_view(0)
        seqlens = torch.tensor([P + 1], device="cuda")
        k_out = paged_mla_decode_attn(
            q_full, kv, bt, seqlens, attn.scale, attn.kv_lora, num_splits=1
        )
        r_out = paged_mla_decode_attn_ref(q_full, kv, bt, seqlens, attn.scale, attn.kv_lora)
    print(
        "decode kernel vs ref: max|d| =",
        float((k_out - r_out).abs().max()),
        "mean|d| =",
        float((k_out - r_out).abs().mean()),
    )


if __name__ == "__main__":
    main()

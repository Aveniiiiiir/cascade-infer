"""检查非连续输入对 rope 的影响 + 打印真实投影值。"""

from __future__ import annotations

import os

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")


def main() -> None:
    from cascade.model_executor.model_loader import load_deepseek_v2

    model, _cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    a = model.model.layers[0].self_attn
    rope = model.model.rope
    x = (torch.arange(6 * 2048, dtype=torch.float32).reshape(6, 2048) / 1000.0 - 3.0)
    x = x.to(torch.bfloat16).cuda()
    pos = torch.arange(6, device="cuda")
    with torch.no_grad():
        qn, qp = a.project_q(x)
        c, kpe = a.project_kv(x)
        qp2a, kpe2a = rope.apply(qp, kpe, pos)  # 非连续 view
        qp2b, kpe2b = rope.apply(qp.contiguous(), kpe.contiguous(), pos)
        cos, sin = rope.cos_sin(6)
    print("qp contiguous?", qp.is_contiguous(), "| kpe contiguous?", kpe.is_contiguous())
    print("rope noncontig vs contig: q max|d|", float((qp2a - qp2b).abs().max()),
          "k max|d|", float((kpe2a - kpe2b).abs().max()))
    print("qp[0,0,:4]", qp[0, 0, :4].tolist())
    print("qp2a[0,0,:4]", qp2a[0, 0, :4].tolist())
    print("qp2b[0,0,:4]", qp2b[0, 0, :4].tolist())
    print("cos[0,:4]", cos[0, :4].tolist(), "sin[0,:4]", sin[0, :4].tolist())
    print("cos[1,:4]", cos[1, :4].tolist(), "sin[1,:4]", sin[1, :4].tolist())
    # 手算第一行 rope 对照
    half = 64 // 2
    q0 = qp[0, 0].float()
    h = torch.cat((-q0[half:], q0[:half]))
    ref0 = q0 * cos[0].float() + h * sin[0].float()
    print("manual ref[0,:4]", ref0[:4].tolist())


if __name__ == "__main__":
    main()

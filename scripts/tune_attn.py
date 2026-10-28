"""Launch settings of the flash-decode attention kernel: time per call over a grid, at decode-like shapes.

    python scripts/tune_attn.py --out tune.json
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from triton.testing import do_bench

from omniasr_trt import kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    h, d, cap = 8, 512, 3072
    report = {}
    for b in (1, 4, 8, 16):
        k = torch.randn(16, h, cap, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
        lengths = torch.full((b,), 1848, device="cuda")
        ref = kernels.decode_attention(q, k, v, lengths, 2048, d**-0.5, 16, 4, 1, 264).float()
        gb = b * h * 1848 * d * 2 * 2 / 1e9
        rows = []
        for block, warps, stages, programs in itertools.product((16, 32, 64), (4, 8), (1, 2, 3, 4),
                                                                 (264, 528, 1056)):
            try:
                out = kernels.decode_attention(q, k, v, lengths, 2048, d**-0.5, block, warps, stages, programs)
                assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
                ms = do_bench(lambda: kernels.decode_attention(q, k, v, lengths, 2048, d**-0.5, block, warps,
                                                               stages, programs), warmup=20, rep=100)
            except Exception as e:  # e.g. out of shared memory
                rows.append({"block": block, "warps": warps, "stages": stages, "programs": programs,
                             "error": str(e)[:80]})
                continue
            rows.append({"block": block, "warps": warps, "stages": stages, "programs": programs, "us": ms * 1000,
                         "tb_per_s": gb / ms})
        ok = sorted((r for r in rows if "us" in r), key=lambda r: r["us"])
        report[b] = {"best": ok[:5], "default": next(r for r in rows if r["block"] == 16 and r["warps"] == 4
                                                    and r["stages"] == 3 and r["programs"] == 264)}
        print(b, json.dumps(report[b]), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

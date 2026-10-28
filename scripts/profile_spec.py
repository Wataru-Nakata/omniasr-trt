"""Wall time of one graphed decode step: one token per row against drafted steps of k + 1 tokens; ``--tune``:
launch settings of ``kernels.draft_attention``, ``--breakdown``: kernel times of eager steps.

    python scripts/profile_spec.py --weights llm-bf16.safetensors --rows 8 --keys 2048 --out spec-step.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

def timed(dec, b, keys, first, reps=50):
    def reset():
        dec.tok[:b] = first
        dec.last[:b] = keys - 600
        dec.done[:b] = False

    reset()
    run = dec.runner(b, keys)
    for _ in range(5):
        reset()
        run()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(reps):
        reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    return total / reps * 1000


def breakdown(dec, b, keys, first, label):
    """Kernel times of one eager step (CUDA time per kernel name, top 25)."""
    from torch.profiler import ProfilerActivity, profile

    dec.tok[:b] = first
    dec.last[:b] = keys - 600
    dec.done[:b] = False
    dec.step(b, keys)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            dec.tok[:b] = first
            dec.last[:b] = keys - 600
            dec.done[:b] = False
            dec.step(b, keys)
        torch.cuda.synchronize()
    rows = sorted(((e.key[:80], e.count / 10, e.self_device_time_total / 10 / 1000) for e in prof.key_averages()
                   if e.device_type.name == "CUDA" and e.self_device_time_total > 0), key=lambda r: -r[2])
    print(f"{label}: {sum(r[2] for r in rows):.3f} ms of kernels, {sum(r[1] for r in rows):.0f} launches")
    for k, c, t in rows[:25]:
        print(f"  {t:.4f} ms  x{c:.0f}  {k}")


def tune(b, keys, length, report):
    """draft_attention launch settings (T = 5 and 9 queries per row) against decode_attention (one query)."""
    import itertools

    from omniasr_trt import kernels

    h, d = 8, 512
    kc = torch.randn(b, h, keys, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    lengths = torch.full((b,), length, device="cuda")

    def ms(fn, reps=200):
        for _ in range(10):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    q1 = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
    report["decode_attention"] = ms(lambda: kernels.decode_attention(q1, kc, vc, lengths, keys, d**-0.5))
    print(f"decode_attention b{b} {length} keys: {report['decode_attention'] * 1000:.1f} us", flush=True)
    for t in (5, 9):
        q = torch.randn(b * t, h, d, device="cuda", dtype=torch.bfloat16)
        results = []
        for block, warps, stages, programs in itertools.product((32, 64), (4, 8), (1, 2, 3), (264, 528, 1056)):
            cfg = {"block": block, "warps": warps, "stages": stages, "programs": programs}
            try:
                results.append((ms(lambda: kernels.draft_attention(q, kc, vc, lengths, t, keys, d**-0.5, **cfg)), cfg))
            except Exception as e:  # shared memory
                print("skip", cfg, type(e).__name__)
        results.sort(key=lambda r: r[0])
        report[f"draft_attention_t{t}"] = [{"us": r[0] * 1000, **r[1]} for r in results]
        for us, cfg in results[:6]:
            print(f"draft_attention T={t}: {us * 1000:.1f} us {cfg}", flush=True)
        default = [r for r in results if r[1] == kernels.DRAFT_ATTN]
        if default:
            print(f"  default {default[0][0] * 1000:.1f} us", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="the LLM's .safetensors (or .pt)")
    p.add_argument("--rows", type=int, nargs="+", default=[8])
    p.add_argument("--keys", type=int, default=2048)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    p.add_argument("--breakdown", action="store_true", help="kernel times of eager steps (greedy, k = 4)")
    p.add_argument("--tune", action="store_true", help="draft_attention launch settings first")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from omniasr_trt.decode import GreedyDecoder
    from omniasr_trt.modeling import load_checkpoint
    from omniasr_trt.spec import SpecDecoder

    m = load_checkpoint("llm", args.weights, torch.bfloat16)
    report = {}
    if args.tune:
        report["tune"] = {}
        tune(8, args.keys, args.keys - 600, report["tune"])
    with torch.inference_mode():
        for b in args.rows:
            ctx = [torch.randn(args.keys - 600, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(b)]
            dec = GreedyDecoder(m, max_batch=b)
            dec.ensure_capacity(args.keys + 256)
            first = dec.prefill(ctx)
            row = {"greedy": timed(dec, b, args.keys, first)}
            if args.breakdown:
                breakdown(dec, b, args.keys, first, f"greedy b{b}")
            dec.release()
            for k in args.k:
                dec = SpecDecoder(m, draft_tokens=k, max_batch=b)
                dec.ensure_capacity(args.keys + 256)
                first = dec.prefill(ctx)
                dec.first[:b] = args.keys - 600
                row[f"k{k}"] = timed(dec, b, args.keys, first)
                if args.breakdown and k == 4:
                    breakdown(dec, b, args.keys, first, f"drafted k{k} b{b}")
                dec.release()
            report[b] = row
            print(b, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

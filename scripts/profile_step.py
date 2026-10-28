"""Kernel time breakdown of one LLM decode step (eager, fused kernels) and the graphed step's wall time.

    python scripts/profile_step.py --weights llm-bf16.safetensors --rows 1 8 --out step.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    p.add_argument("--keys", type=int, default=2048)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from omniasr_trt.decode import GreedyDecoder
    from omniasr_trt.modeling import build, load_checkpoint

    m = load_checkpoint(build("llm"), args.weights, torch.bfloat16)
    dec = GreedyDecoder(m, max_batch=max(args.rows), sync_every=32)
    dec.ensure_capacity(args.keys + 256)
    report = {}
    with torch.inference_mode():
        for b in args.rows:
            ctx = [torch.randn(args.keys - 200, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(b)]
            first = dec.prefill(ctx)

            def reset():
                dec.tok[:b] = first
                dec.last[:b] = args.keys - 201
                dec.done[:b] = False

            reset()
            run = dec.runner(b, args.keys)
            reset()
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                run()
            torch.cuda.synchronize()
            graphed = (time.perf_counter() - t0) / 50
            reset()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    dec.step(b, args.keys)
                torch.cuda.synchronize()
            rows = []
            for e in prof.key_averages():
                if e.device_type.name == "CUDA" and e.self_device_time_total > 0:
                    rows.append((e.key[:90], e.count / 10, e.self_device_time_total / 10 / 1000))
            rows.sort(key=lambda r: -r[2])
            kernel_ms = sum(r[2] for r in rows)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                dec.prefill(ctx)
                torch.cuda.synchronize()
            pre = sorted(((e.key[:90], e.count, e.self_device_time_total / 1000) for e in prof.key_averages()
                          if e.device_type.name == "CUDA" and e.self_device_time_total > 0), key=lambda r: -r[2])
            print(f"prefill {b} x {args.keys - 200}: {sum(r[2] for r in pre):.2f} ms of kernels")
            for k, c, t in pre[:8]:
                print(f"  {t:.3f} ms  x{c}  {k}")
            report[b] = {"graphed_step_ms": graphed * 1000, "kernel_sum_ms": kernel_ms,
                         "prefill_kernel_ms": sum(r[2] for r in pre),
                         "kernels_per_step": sum(r[1] for r in rows),
                         "top": [{"kernel": k, "per_step": c, "ms": round(t, 4)} for k, c, t in rows[:25]]}
            print(json.dumps({k: v for k, v in report[b].items() if k != "top"}), flush=True)
            for r in report[b]["top"]:
                print(f"  {r['ms']:.4f} ms  x{r['per_step']:.0f}  {r['kernel']}")
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Speed and transcripts of omniasr-trt on a manifest of audio files.

    python scripts/bench.py llm --weights llm-bf16.safetensors --tokenizer omniASR_tokenizer_written_v2.model \\
        --manifest data.jsonl --batch 8 --repeat 2 --out ours-llm.json
    python scripts/bench.py ctc --weights ctc-bf16.safetensors --tokenizer ... --manifest data.jsonl --out ours-ctc.json

Manifest: one JSON object per line, ``{"id": ..., "audio": "path.wav", "lang": "ben_Beng"}`` (``lang`` optional:
the LLM's language condition, null = unconditioned). Inputs are processed longest first in calls of ``--batch``; a
last 15 s piece shorter than 0.5 s is padded with silence to 0.5 s (omnilingual-asr cannot encode a few ms), the
same in ``tools/reference.py``. Every pass is timed; the first one includes the Triton compilation and the CUDA
graph captures.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omniasr_trt.cli import load_wave  # noqa: E402


def tail_padding(samples: int, segment: int = 240000, minimum: int = 8000) -> int:
    tail = samples % segment
    return minimum - tail if 0 < tail < minimum else 0


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=["llm", "ctc"])
    p.add_argument("--weights", required=True, help=".safetensors from `omniasr-trt convert`, or the official .pt")
    p.add_argument("--tokenizer")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch", type=int, default=8, help="inputs per call")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--beam-size", type=int, default=1, help="LLM: omnilingual-asr's beam search (1: greedy)")
    p.add_argument("--length-norm", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from omniasr_trt import OmniASRCTC, OmniASRLLM

    items = read_manifest(args.manifest)
    waves = [load_wave(it["audio"]) for it in items]
    waves = [np.pad(w, (0, tail_padding(len(w)))) for w in waves]
    langs = [it.get("lang") for it in items]
    dtype = getattr(torch, args.dtype)
    t0 = time.monotonic()
    if args.model == "llm":
        model = OmniASRLLM.from_pretrained(args.weights, args.tokenizer, dtype=dtype, max_batch=args.batch,
                                           beam_size=args.beam_size, length_norm=args.length_norm)
    else:
        model = OmniASRCTC.from_pretrained(args.weights, args.tokenizer, dtype=dtype)
    load = time.monotonic() - t0
    order = sorted(range(len(waves)), key=lambda i: -len(waves[i]))
    seconds = sum(len(w) for w in waves) / 16000
    passes = []
    for _ in range(args.repeat):
        texts = [None] * len(waves)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for k in range(0, len(order), args.batch):
            chosen = order[k : k + args.batch]
            batch = [waves[i] for i in chosen]
            outs = model.transcribe(batch, [langs[i] for i in chosen]) if args.model == "llm" else \
                model.transcribe(batch)
            for i, text in zip(chosen, outs, strict=True):
                texts[i] = text
        torch.cuda.synchronize()
        wall = time.monotonic() - started
        passes.append({"wall_seconds": wall, "rtfx": seconds / wall,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        print(json.dumps(passes[-1]), flush=True)
    result = {"impl": "omniasr-trt", "model": args.model, "dtype": args.dtype, "batch": args.batch,
              "beam_size": args.beam_size, "length_norm": args.length_norm,
              "inputs": len(waves), "audio_seconds": seconds, "load_seconds": load, "passes": passes,
              "ids": [it.get("id", it["audio"]) for it in items], "texts": texts}
    args.out.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
